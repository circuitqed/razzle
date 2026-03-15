# Razzle Dazzle Engine
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN mkdir -p razzle && echo '__version__ = "0.1.0"' > razzle/__init__.py \
    && pip install --no-cache-dir -e ".[server]"

# Copy actual source code (changes here won't trigger pip install)
COPY razzle/ razzle/
COPY server/ server/
COPY cli/ cli/
COPY scripts/ scripts/
COPY data/ data/

# Expose port
EXPOSE 8000

# Default command - run server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
