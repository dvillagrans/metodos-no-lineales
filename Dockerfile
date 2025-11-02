# Production container for the Flask app
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable output flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

# Install system deps if needed in the future
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 5000

# Use gunicorn to serve the Flask app
# app:app refers to the `app` object inside app.py
CMD ["gunicorn", "-k", "gthread", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
