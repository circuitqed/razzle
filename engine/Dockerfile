# Razzle Dazzle Engine
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml .
COPY razzle/ razzle/
COPY server/ server/
COPY cli/ cli/
COPY scripts/ scripts/

# Install package with server dependencies
RUN pip install --no-cache-dir -e ".[server]"

# Expose port
EXPOSE 8000

# Default command - run server with auto-reload
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
