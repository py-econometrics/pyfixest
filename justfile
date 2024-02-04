# docs: justfile docs: https://just.systems/man/en/
default:
	just --list --unsorted

# Clean environment
[confirm]
clean:
	rm -rf .venv/

# Update dependencies in pyproject.toml
poetry-update:
	poetry update

# Install development dependencies
install-dev:
	poetry env use 3.11
	poetry config virtualenvs.in-project true
	poetry install

# Install R dependencies
install-r:
	Rscript -e 'install.packages(c("broom", "clubSandwich", "did2s", "fixest"), repos="https://cran.rstudio.com")'

# Run pytest
tests:
	poetry run pytest

# Build the package
build: tests
	poetry build

# Build documentation and website
docs-build:
	poetry run quartodoc build --verbose --config docs/_quarto.yml

# Render documentation and website
render: docs-build
	poetry run quarto render docs

# Build the documentation and watch for changes
docs-watch:
	poetry run poetry run quartodoc build --watch --verbose --config docs/_quarto.yml

# Preview the docs
preview:
	poetry run quarto preview docs

# Clean docs build
docs-clean:
	rm -rf docs/_build docs/api/api_card
