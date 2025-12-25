FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install system dependencies needed for building Python packages and Rust extensions
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install pixi for the vscode user
USER vscode
ENV PIXI_HOME=/home/vscode/.pixi
ENV PATH="${PIXI_HOME}/bin:${PATH}"

RUN curl -fsSL https://pixi.sh/install.sh | bash

# Set working directory
WORKDIR /workspaces/pyfixest
