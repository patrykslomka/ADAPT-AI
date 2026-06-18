FROM python:3.12-slim

WORKDIR /app

# Install system deps for chromadb / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for layer caching
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Ollama runs on the host - point to host.docker.internal
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1

# Default: run the test suite (no live calls needed)
CMD ["pytest", "tests/test_adapt_ai/", "-q"]
