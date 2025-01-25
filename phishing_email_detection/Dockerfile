# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-install numpy to avoid conflicts
RUN pip install --upgrade pip && pip install numpy==1.23.5

# Copy project files into the container
COPY . /app

# Install remaining dependencies
RUN pip install -r requirements.txt

# Ensure model files are included in the container
COPY models/phishing_model.pkl /app/models/
COPY models/tfidf_vectorizer.pkl /app/models/

# Expose the port Flask will run on
EXPOSE 5001

# Run the Flask app using Gunicorn for better performance
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
