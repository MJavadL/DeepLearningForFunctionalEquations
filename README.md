# Functional Equation Neural Solver

A small PyTorch project that trains a neural network to approximate solutions of the functional equation:

```text
f(f(x)) - f(x) = x
```

on a bounded interval. The training loop minimizes the residual

```text
R(x) = f(f(x)) - f(x) - x
```

and adds a sign-consistency penalty so the learned function tends to preserve the sign of the input.

## Repository layout

```text
functional-eq-neural-solver/
├── train_functional_eq.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

## Features

- Configurable training interval, network depth, width, batch size, learning rate, and seed
- Automatic CPU/GPU selection
- Saves trained weights to `output/functional_eq_model.pth`
- Saves a summary plot to `output/functional_eq_summary.png`
- Compares the learned solution to the two known affine branches `phi*x` and `psi*x`

## Requirements

- Python 3.10+
- PyTorch
- Matplotlib

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate, 
                            # in case of  PowerShell execution policy issue use the following:
                            # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
                            # .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run with default parameters:

```bash
python train_functional_eq.py
```

Run with custom settings:

```bash
python train_functional_eq.py --epochs 3000 --batch-size 512 --lr 1e-3 --show
```

All generated artifacts are written to the `output/` directory by default.
Example for the given functional equation and default hyperparameters as follows:

<img width="1900" height="616" alt="functional_eq_summary" src="https://github.com/user-attachments/assets/8440af74-3b5d-4566-9a30-af4ca2197af5" />


## Example output

During training, the script prints the total loss and residual loss every 500 epochs. At the end it reports:

- MSE to `phi * x`
- MSE to `psi * x`
- Estimated slope from a least-squares fit through the origin

## Notes

- The current objective uses the residual loss plus a sign-consistency penalty.
- The original anchoring and weak output regularization terms can be reintroduced if you want stronger control of the solution family.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
