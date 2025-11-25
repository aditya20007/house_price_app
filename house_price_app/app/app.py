import os
import sys
import traceback
from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    return app

app = create_app()

try:
    from src.utils import load_model, predict_price
    model = load_model()
except Exception as e:
    print("Failed to load model:", e)
    traceback.print_exc()
    try:
        auto_train = os.getenv("AUTO_TRAIN", "1") == "1"
        if auto_train:
            from src.preprocess import ensure_raw, clean_save
            from src.train import main as train_main
            ensure_raw(force=False, country=os.getenv("COUNTRY", "IN"))
            clean_save()
            train_main()
            model = load_model()
        else:
            model = None
    except Exception as ee:
        print("Auto-train failed:", ee)
        traceback.print_exc()
        model = None

@app.route("/")
def index():
    return render_template("index.html", price=None, error=None)

@app.route("/predict", methods=["POST"]) 
def predict():
    try:
        if request.is_json:
            data = request.get_json(force=True)
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
        if not location:
            raise ValueError("Location is required")
        if model is None:
            msg = "Model not loaded. Train first: python house_price_app/src/train.py"
            if request.is_json:
                return jsonify({"error": msg}), 500
            return render_template("index.html", price=None, error=msg)
        features = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "year_built": year_built
        }
        y = predict_price(model, features)
        if request.is_json:
            return jsonify({"price": round(y, 2), "currency": "INR"})
        return render_template("index.html", price=round(y, 2), error=None)
    except Exception as e:
        traceback.print_exc()
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template("index.html", price=None, error=str(e))

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
