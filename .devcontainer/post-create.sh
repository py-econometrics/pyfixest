#!/bin/bash
set -e

echo "=== pyfixest Codespace Setup ==="
echo ""

# Ensure pixi is in PATH
export PATH="$HOME/.pixi/bin:$PATH"

echo "Step 1/3: Installing pixi environments..."
pixi install

echo ""
echo "Step 2/3: Setting up pre-commit hooks via prek..."
pixi run -e lint prek install --install-hooks

echo ""
echo "Step 3/3: Verifying setup..."
pixi run python -c "import pyfixest; print(f'pyfixest {pyfixest.__version__} loaded successfully!')"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Getting started:"
echo "  - Run tests: pixi run test-py"
echo "  - Run linting: pixi run lint"
echo "  - Build docs: pixi run docs-build"
echo ""
echo "See all available tasks: pixi task list"
echo ""
