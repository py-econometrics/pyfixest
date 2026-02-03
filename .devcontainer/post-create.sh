#!/bin/bash
set -e

echo "=== pyfixest Codespace Setup ==="
echo ""

# Ensure pixi is in PATH
export PATH="$HOME/.pixi/bin:$PATH"

echo "Step 1/4: Installing pixi environments..."
pixi install

echo ""
echo "Step 2/4: Compiling Rust extension..."
pixi run -e dev maturin-develop

echo ""
echo "Step 3/4: Setting up pre-commit hooks..."
pixi run -e lint pre-commit install --install-hooks

echo ""
echo "Step 4/4: Verifying setup..."
pixi run -e dev python -c "import pyfixest; print(f'pyfixest {pyfixest.__version__} loaded successfully!')"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Getting started:"
echo "  - Activate dev environment: pixi shell -e dev"
echo "  - Run tests: pixi run tests-regular"
echo "  - Run all tests (including R): pixi run tests"
echo "  - Run linting: pixi run -e lint pre-commit run --all-files"
echo "  - Build docs: pixi run -e docs docs-build"
echo ""
echo "Available pixi environments:"
echo "  - dev: Development with testing and R"
echo "  - docs: Documentation building"
echo "  - lint: Code linting"
echo "  - build: Building packages"
echo ""
