# Use official Python image as base
FROM python:3.11-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --fix-missing gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose port for API
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "api.frontend_routing.app:app", "--host", "0.0.0.0", "--port", "8000"]
