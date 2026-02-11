from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from datetime import datetime

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# Flask App Setup

app = Flask(__name__)


# Load Trained Model


model = pickle.load(open("RidgeModel.pkl", "rb"))

# Extract known locations from OneHotEncoder
encoder = model.named_steps['columntransformer'] \
               .named_transformers_['onehotencoder']
known_locations = list(encoder.categories_[0])


# API Key Configuration


VALID_API_KEYS = {
    "demo-key-123": "public-demo-user",
    "client-key-456": "mobile-app-client"
}


# Rate Limiter Setup
# Limits per API key if provided,
# otherwise fallback to IP address


limiter = Limiter(
    key_func=lambda: (
        request.get_json().get("api_key")
        if request.is_json and request.get_json()
        else get_remote_address()
    ),
    app=app,
    default_limits=["100 per day", "50 per hour"]
)

# Request Logging

@app.before_request
def log_request():
    print({
        "time": datetime.now().isoformat(),
        "ip": request.remote_addr,
        "path": request.path,
        "method": request.method
    })


# FRONTEND ROUTES


@app.route("/")
def home():
    return render_template("index.html", locations=known_locations)

@app.route("/predict", methods=["POST"])
def predict_form():

    location = request.form["location"].strip()
    total_sqft = float(request.form["total_sqft"])
    bath = int(request.form["bath"])
    bhk = int(request.form["bhk"])

    # Handle unknown locations
    if location not in known_locations:
        location = "other"

    input_df = pd.DataFrame([{
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "bhk": bhk
    }])

    prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2),
        locations=known_locations
    )


# REST API ROUTE


@app.route("/api/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict_api():

    data = request.get_json()

    #  API Key Validation 

    api_key = data.get("api_key")

    if not api_key or api_key not in VALID_API_KEYS:
        return jsonify({
            "error": "Unauthorized access",
            "message": "Valid API key required"
        }), 401

    required_fields = ["location", "total_sqft", "bath", "bhk"]

    for field in required_fields:
        if field not in data:
            return jsonify({
                "error": f"Missing field: {field}"
            }), 400

    location = data["location"].strip()
    total_sqft = float(data["total_sqft"])
    bath = int(data["bath"])
    bhk = int(data["bhk"])

    if location not in known_locations:
        location = "other"

    input_df = pd.DataFrame([{
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "bhk": bhk
    }])

    prediction = model.predict(input_df)[0]

    print(f"API used by: {VALID_API_KEYS[api_key]} from {request.remote_addr}")

    return jsonify({
        "location_used": location,
        "predicted_price_lakh": round(prediction, 2)
    })


# Custom Rate Limit Error


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please wait before trying again."
    }), 429


# Run App


if __name__ == "__main__":
    app.run(debug=True)
