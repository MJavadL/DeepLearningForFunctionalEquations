"""
Train a small neural network to approximate solutions of the functional equation

    f(f(x)) - f(x) = x

on a bounded interval.

What this script does:
- Trains a neural network f(x) so that the residual
      R(x) = f(f(x)) - f(x) - x
  is minimized on a user-defined interval.
- Adds a sign-consistency regularizer to discourage solutions that flip the sign.
- Compares the learned function to the two known linear solutions:
      f(x) = a x  with  a^2 = a + 1
- Saves model weights and diagnostic plots to disk.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Create and return a command-line argument parser that controls:
    - Training interval
    - Neural network architecture
    - Optimizer hyperparameters
    - Output and visualization options
    """
    parser = argparse.ArgumentParser(
        description="Train a neural network for the functional equation f(f(x)) - f(x) = x."
    )

    # Training domain
    parser.add_argument("--xmin", type=float, default=-2.0, help="Lower bound of the training interval.")
    parser.add_argument("--xmax", type=float, default=2.0, help="Upper bound of the training interval.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size used for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")

    # Network architecture
    parser.add_argument("--width", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--depth", type=int, default=3, help="Number of hidden layers.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Regularization control
    parser.add_argument(
        "--sign-margin",
        type=float,
        default=1e-3,
        help="Margin used in the sign-consistency penalty.",
    )

    # Output control
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where model weights and plots are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )

    return parser


# ------------------------------------------------------------------------------
# Neural network definition
# ------------------------------------------------------------------------------

class FuncNet(nn.Module):
    """
    Fully connected neural network representing the scalar function f(x).

    Architecture:
    - Input: scalar x
    - Several hidden layers with Tanh activation
    - Output: scalar f(x)
    """

    def __init__(self, width: int = 64, depth: int = 3) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = 1  # One-dimensional input

        # Construct hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width

        # Final output layer (no activation)
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute f(x)."""
        return self.net(x)


# ------------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------------

def sign_constraint_loss(
    model: nn.Module,
    x: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Penalize outputs whose sign disagrees with the sign of x.

    The penalty enforces:
        f(x) * x >= margin

    This discourages pathological solutions such as f(x) ≈ -x.
    """
    fx = model(x)
    prod = fx * x
    return torch.mean(torch.relu(margin - prod) ** 2)


def residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the functional-equation residual:

        R(x) = f(f(x)) - f(x) - x
    """
    fx = model(x)
    ffx = model(fx)
    return ffx - fx - x


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------

def train(model: nn.Module, args: argparse.Namespace, device: torch.device) -> list[float]:
    """
    Train the network by minimizing:
        mean(R(x)^2) + sign-consistency penalty
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Uniform sampling from the training interval
        x = args.xmin + (args.xmax - args.xmin) * torch.rand(
            args.batch_size, 1, device=device
        )

        # Functional equation loss
        r = residual(model, x)
        loss_fe = torch.mean(r ** 2)

        # Sign constraint loss
        loss_sign = sign_constraint_loss(model, x, margin=args.sign_margin)

        # Total loss, if you add loss_sign term here, the solution with positive slope is obtained
        loss = loss_fe # + loss_sign

        # Gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Progress logging
        if epoch % 500 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | total loss = {loss.item():.6e} | "
                f"residual loss = {loss_fe.item():.6e}"
            )

    return loss_history


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    xmin: float,
    xmax: float,
    device: torch.device,
) -> dict[str, float | torch.Tensor]:
    """
    Evaluate the trained model on a dense grid and compare it to
    known analytic solutions.
    """
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(xmin, xmax, 500, device=device).view(-1, 1)
        y_plot = model(x_plot)
        r_plot = residual(model, x_plot)

    # Convert tensors to NumPy for plotting
    x_np = x_plot.cpu().numpy().flatten()
    y_np = y_plot.cpu().numpy().flatten()
    r_np = r_plot.cpu().numpy().flatten()

    # Solutions of a^2 = a + 1
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2

    y_phi = phi * x_np
    y_psi = psi * x_np

    # Mean squared errors against exact linear solutions
    mse_phi = ((y_np - y_phi) ** 2).mean()
    mse_psi = ((y_np - y_psi) ** 2).mean()

    # Best linear slope fit (least squares)
    slope_fit = (x_np @ y_np) / (x_np @ x_np)

    return {
        "x_np": x_np,
        "y_np": y_np,
        "r_np": r_np,
        "y_phi": y_phi,
        "y_psi": y_psi,
        "phi": phi,
        "psi": psi,
        "mse_phi": float(mse_phi),
        "mse_psi": float(mse_psi),
        "slope_fit": float(slope_fit),
    }


# ------------------------------------------------------------------------------
# Saving plots and files
# ------------------------------------------------------------------------------

def save_artifacts(
    loss_history: list[float],
    eval_data: dict[str, float | torch.Tensor],
    output_dir: Path,
    show: bool,
) -> None:
    """
    Save training diagnostics:
    - Loss history
    - Learned function vs analytic solutions
    - Residual plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))

    # Training loss
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.yscale("log")
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Learned function
    plt.subplot(1, 3, 2)
    plt.plot(eval_data["x_np"], eval_data["y_np"], label="Learned f(x)", linewidth=2)
    plt.plot(eval_data["x_np"], eval_data["y_phi"], "--", label=r"$\phi x$")
    plt.plot(eval_data["x_np"], eval_data["y_psi"], "--", label=r"$\psi x$")
    plt.title("Learned function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    # Residual
    plt.subplot(1, 3, 3)
    plt.plot(eval_data["x_np"], eval_data["r_np"], label="Residual R(x)")
    plt.title("Residual")
    plt.xlabel("x")
    plt.ylabel("R(x)")
    plt.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "functional_eq_summary.png", dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Main execution pipeline:
    - Parse arguments
    - Train model
    - Save weights and plots
    """
    args = build_parser().parse_args()

    # Select CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"Using device: {device}")

    # Initialize model
    model = FuncNet(width=args.width, depth=args.depth).to(device)

    # Train
    loss_history = train(model, args, device)

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "functional_eq_model.pth"
    torch.save(model.state_dict(), model_path)

    # Evaluate and visualize
    eval_data = evaluate(model, args.xmin, args.xmax, device)
    save_artifacts(loss_history, eval_data, args.output_dir, args.show)

    # Summary
    print("Comparison with known linear solutions:")
    print(f"MSE to phi*x (phi={eval_data['phi']:.6f}) = {eval_data['mse_phi']:.6e}")
    print(f"MSE to psi*x (psi={eval_data['psi']:.6f}) = {eval_data['mse_psi']:.6e}")
    print(f"Estimated fitted slope ≈ {eval_data['slope_fit']:.6f}")
    print(f"Saved model weights to: {model_path}")
    print(f"Saved plot to: {args.output_dir / 'functional_eq_summary.png'}")


if __name__ == "__main__":
    main()