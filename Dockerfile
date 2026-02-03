# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies with timeouts
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for better network handling
RUN pip config set global.timeout 1000
RUN pip config set global.retries 3

# Copy requirements and install Python dependencies with extended timeouts
COPY requirements.txt .

# Upgrade pip first and configure timeouts
RUN pip install --upgrade pip

# Install PyTorch CPU-only first (smaller download, no CUDA)
RUN pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies with extended timeouts and retry mechanisms
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 3 \
    --default-timeout=1000 \
    -r requirements.txt

# Copy application code
COPY . .

# Create results directories
RUN mkdir -p results/clustering results/causal results/llm_responses

# Expose ports
EXPOSE 8501 8000

# Set Python path
ENV PYTHONPATH=/app:/app/helpers:/app/LOCI:/app/ROCHE:/app/DATA

# OpenAI API Key (will be passed from docker-compose)
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Make run script executable
RUN chmod +x run.sh

# Start the application
CMD ["./run.sh"]