# Dev container for the Predict Customer Churn project.
# Ubuntu 24.04 + Python 3.12 + uv; all project dependencies (runtime + dev) are
# installed from uv.lock into /opt/venv at build time.

FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# uv/uvx binaries from the official image (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/

# System packages: Python 3.12 (Ubuntu 24.04 default) and fonts for matplotlib
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        fonts-dejavu-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Keep the virtual environment outside the bind-mounted workspace so it never
# clashes with a host .venv (e.g. a Windows one).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    PATH="/opt/venv/bin:${PATH}"

RUN mkdir -p /opt/venv && chown vscode:vscode /opt/venv

USER vscode
WORKDIR /tmp/deps

# Install dependencies only (no project code) so this layer is cached until the lock changes
COPY --chown=vscode:vscode pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --all-groups \
    && rm -rf /tmp/deps/* \
    && uv cache clean

WORKDIR /workspaces
