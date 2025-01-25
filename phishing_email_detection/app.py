from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define paths
MODEL_PATH = "models/phishing_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model_and_vectorizer():
    """Loads the trained phishing detection model and TF-IDF vectorizer."""
    global model, vectorizer

    if not os.path.exists(MODEL_PATH):
        logging.error(f"❌ Model file '{MODEL_PATH}' not found.")
        return
    
    if not os.path.exists(VECTORIZER_PATH):
        logging.error(f"❌ Vectorizer file '{VECTORIZER_PATH}' not found.")
        return

    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info("✅ Model loaded successfully.")

        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        logging.info("✅ Vectorizer loaded successfully.")

    except Exception as e:
        logging.error(f"❌ Error loading model/vectorizer: {e}")

# Load models on startup
load_model_and_vectorizer()

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "running", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for phishing email classification."""
    if not request.json or "email_content" not in request.json:
        return jsonify({"error": "Missing 'email_content' field"}), 400

    email_content = request.json["email_content"]

    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    try:
        transformed_content = vectorizer.transform([email_content])
        prediction = model.predict(transformed_content)[0]  # Assuming binary classification
        prediction_label = "Phishing" if prediction == 1 else "Legitimate"
        return jsonify({"prediction": prediction_label, "status": "success"})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=PORT, debug=True)
