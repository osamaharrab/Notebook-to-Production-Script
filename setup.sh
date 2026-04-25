#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "=== Petra Telecom Churn Model Comparison — Setup ==="

# Create virtual environment if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade build tools
echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install main dependencies
echo "Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

# Optionally install dev dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "Installing dev dependencies from requirements-dev.txt..."
    python -m pip install -r requirements-dev.txt
fi

# Optionally run environment test
if [ -f "test_environment.py" ]; then
    echo "Running test_environment.py..."
    python test_environment.py
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:  source $VENV_DIR/bin/activate"
echo "Then run:  python model_comparison.py --data-path data/telecom_churn.csv --dry-run"
