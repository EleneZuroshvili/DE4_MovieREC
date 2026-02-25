# Base image: Microsoft Dev Containers Python 3.13 (used by GitHub Codespaces)
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# OCI labels for GitHub Container Registry
LABEL org.opencontainers.image.source=https://github.com/zoltanctoth/ceu-ai-engineering-class
LABEL org.opencontainers.image.description="CEU AI Engineering Class Development Container"

# Install AWS CLI v2 (do this early as it rarely changes)
RUN ARCH=$(uname -m) && \
    cd /tmp && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Install build dependencies for pysqlite3 from source
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch to vscode user for UV installation
USER vscode

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

WORKDIR /workspaces/ceu-ai-engineering-class

# Copy project files for dependency installation
COPY --chown=vscode:vscode pyproject.toml uv.lock .python-version ./

# Create virtual environment and install dependencies using UV
# --no-binary pysqlite3 forces source build against system libsqlite3-dev (works on both amd64 and arm64)
RUN uv sync --frozen --no-binary-package pysqlite3
