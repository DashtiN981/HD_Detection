# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install specific version of scikit-learn along with other dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy pydantic xgboost scikit-learn==1.2.1

# Expose port 8000 to allow access to the FastAPI application
EXPOSE 8000

# Run app.py using Uvicorn, with FastAPI, on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]