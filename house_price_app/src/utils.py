import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "house_model.pkl")

def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def predict_price(model, features: dict) -> float:
    df = pd.DataFrame([features], columns=["area", "bedrooms", "bathrooms", "location", "year_built"])
    y = model.predict(df)[0]
    return float(y)
