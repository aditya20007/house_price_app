import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    return app

app = create_app()

try:
    from src.utils import load_model, predict_price
    model = load_model()
except Exception:
    model = None

@app.route("/")
def index():
    return render_template("index.html", price=None)

@app.route("/predict", methods=["POST"]) 
def predict():
    if request.is_json:
        data = request.get_json()
        area = float(data.get("area", 0))
        bedrooms = int(data.get("bedrooms", 0))
        bathrooms = int(data.get("bathrooms", 0))
        location = str(data.get("location", ""))
        year_built = int(data.get("year_built", data.get("year", 2000)))
    else:
        area = float(request.form.get("area", 0))
        bedrooms = int(request.form.get("bedrooms", 0))
        bathrooms = int(request.form.get("bathrooms", 0))
        location = request.form.get("location", "")
        year_built = int(request.form.get("year_built", request.form.get("year", 2000)))
    features = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": location,
        "year_built": year_built
    }
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    y = predict_price(model, features)
    if request.is_json:
        return jsonify({"price": round(y, 2)})
    return render_template("index.html", price=round(y, 2))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
