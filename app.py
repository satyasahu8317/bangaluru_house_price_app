from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML pipeline
model = pickle.load(open("RidgeModel.pkl", "rb"))

# 🔹 Extract known locations from encoder
encoder = model.named_steps['columntransformer'] \
               .named_transformers_['onehotencoder']
locations = sorted(list(encoder.categories_[0]))

@app.route("/")
def home():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    location = request.form["location"]
    total_sqft = float(request.form["total_sqft"])
    bath = int(request.form["bath"])
    bhk = int(request.form["bhk"])

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
        locations=locations
    )

if __name__ == "__main__":
    app.run(debug=True)
