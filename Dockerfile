# Base image Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    joblib

# Copy MLflow Project
COPY MLProject /app/MLProject

# Masuk ke MLProject
WORKDIR /app/MLProject

# Expose MLflow port (optional, aman)
EXPOSE 5000

# Jalankan MLflow Project
CMD ["mlflow", "run", ".", "--env-manager", "local"]
