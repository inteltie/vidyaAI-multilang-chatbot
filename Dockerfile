FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Command to run the application with dynamic worker scaling
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port 8001 --workers ${WORKERS:-$(python3 -c 'import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)')}"]
