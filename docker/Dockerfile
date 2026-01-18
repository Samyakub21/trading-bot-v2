# Use an official Python runtime as a parent image (Slim version for smaller size)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code from the build context (project root)
COPY . .

# Create directory for logs, data, and reports to allow volume mounting with permissions
RUN mkdir -p logs data reports && \
    touch bot.log

# Create a non-root user for security and change ownership
RUN adduser --disabled-password --gecos '' trader && \
    chown -R trader:trader /app

# Switch to the non-root user
USER trader

# Run the bot
CMD ["python", "/app/Tradebot.py"]