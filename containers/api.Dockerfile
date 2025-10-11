# Use official Python image as base
FROM python:3.11-bookworm 

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

COPY api app/api
COPY requirements.txt /app/

# Install system dependencies
RUN for i in 1 2 3 4 5; do \
      apt-get update --allow-releaseinfo-change && \
      apt-get install -y --fix-missing gcc cmake build-essential libsentencepiece-dev && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/* && break || sleep 5; \
    done

# Install Python dependencies

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose port for API
EXPOSE 8000

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
