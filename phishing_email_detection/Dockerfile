
# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the app files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the app port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
