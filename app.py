from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML pipeline
model = pickle.load(open("RidgeModel.pkl", "rb"))

# Extract known locations from encoder
encoder = model.named_steps['columntransformer'] \
               .named_transformers_['onehotencoder']
known_locations = list(encoder.categories_[0])

@app.route("/")
def home():
    return render_template("index.html", locations=known_locations)

# ---------- FRONTEND FORM PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict_form():
    location = request.form["location"].strip()
    total_sqft = float(request.form["total_sqft"])
    bath = int(request.form["bath"])
    bhk = int(request.form["bhk"])

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

# ---------- REST API (JSON) ----------
@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    required_fields = ["location", "total_sqft", "bath", "bhk"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

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

    return jsonify({
        "location_used": location,
        "predicted_price_lakh": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
