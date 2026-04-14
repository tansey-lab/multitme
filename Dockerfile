FROM bitnami/pytorch:latest

# Switch to root for installation
USER root

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install the package (cache mount keeps uv cache out of image layers)
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --no-editable

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POLARS_SKIP_CPU_CHECK=1
ENV TRITON_CACHE_DIR=/tmp/triton_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV WANDB_CACHE_DIR=/tmp/wandb_cache
ENV WANDB_DATA_DIR=/tmp/wandb_data
ENV TORCH_HOME=/tmp/torch
ENV PATH="/app/.venv/bin:$PATH"

# Switch back to non-root user
USER 1001
