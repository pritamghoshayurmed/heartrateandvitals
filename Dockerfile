FROM python:3.10-slim

WORKDIR /app

# Install system dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
