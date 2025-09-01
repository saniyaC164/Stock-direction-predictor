# syntax=docker/dockerfile:1

# Use Python 3.11 instead of 3.12 for better package compatibility
ARG PYTHON_VERSION=3.11.8
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy requirements first for better caching
COPY requirements-docker.txt requirements.txt

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download dependencies as a separate step to take advantage of Docker's caching.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY --chown=appuser:appuser . .

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application with proper Streamlit configuration for Docker
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]