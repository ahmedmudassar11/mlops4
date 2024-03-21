# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install joblib module


# Copy the Flask application code and the trained model file into the container
COPY app.py .
# Expose the port on which the Flask app will run (change this if needed)
EXPOSE 5000

# Set the entry point command to run the Flask app
CMD ["python", "app.py"]
