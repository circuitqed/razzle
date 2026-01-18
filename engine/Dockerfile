# Razzle Dazzle Engine - GPU Training Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY razzle/ razzle/
COPY cli/ cli/
COPY scripts/ scripts/

# Install package
RUN pip install --no-cache-dir -e ".[training]"

# Default command
CMD ["python", "-c", "from razzle import __version__; print(f'Razzle Dazzle Engine v{__version__}')"]
