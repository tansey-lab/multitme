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

# Install the package
RUN uv sync --no-dev --no-editable

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POLARS_SKIP_CPU_CHECK=1
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# Switch back to non-root user
USER 1001

# Entry point
ENTRYPOINT ["uv", "run"]
CMD ["multitme-train", "--help"]
