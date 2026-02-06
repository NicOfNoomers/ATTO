#!/bin/bash
# ATTO Pump Test Dashboard - Installation Script (Linux/Mac)
# This script installs all required dependencies for the Pump Testing Suite

set -e

echo "============================================"
echo "ATTO Pump Test Dashboard - Installation"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3.8+ from https://python.org or via your package manager"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check for virtual environment
USE_VENV=0
VENV_PATH=""

if [ -d ".venv/bin/activate" ]; then
    VENV_PATH=".venv"
    USE_VENV=1
elif [ -d "venv/bin/activate" ]; then
    VENV_PATH="venv"
    USE_VENV=1
fi

if [ "$USE_VENV" -eq 1 ]; then
    echo "Virtual environment found at $VENV_PATH"
    echo ""
    read -p "Use existing virtual environment? (Y/n): " USE_VENV_CHOICE
    if [ "$USE_VENV_CHOICE" != "n" ] && [ "$USE_VENV_CHOICE" != "N" ]; then
        echo "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
        USE_VENV=2
    fi
fi

if [ "$USE_VENV" -ne 2 ]; then
    echo ""
    echo "Creating new virtual environment..."
    python3 -m venv .venv
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Upgrading pip..."
    pip install --upgrade pip
fi

echo ""
echo "Installing dependencies from requirements.txt..."
echo ""
pip install -r requirements.txt
echo ""

# Install Fluigent SDK if available
if [ -d "fluigent_sdk-23.0.0" ]; then
    echo "Installing Fluigent SDK..."
    pip install ./fluigent_sdk-23.0.0
    echo ""
fi

echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "To run the dashboard:"
echo "  python pump_test_dashboard.py"
echo ""
echo "For help:"
echo "  python pump_test_dashboard.py --help"
echo ""
