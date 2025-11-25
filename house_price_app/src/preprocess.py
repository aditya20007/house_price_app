import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "clean.csv")

def generate_synthetic(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    locations = ["Springfield", "Riverside", "Greenville", "Hill Valley", "Shelbyville"]
    area = rng.integers(600, 5000, size=n)
    bedrooms = rng.integers(1, 6, size=n)
    bathrooms = rng.integers(1, 4, size=n)
    years = rng.integers(1955, 2024, size=n)
    loc = rng.choice(locations, size=n, replace=True)
    loc_base = {"Springfield": 190000, "Riverside": 220000, "Greenville": 260000, "Hill Valley": 240000, "Shelbyville": 170000}
    rate = {"Springfield": 120, "Riverside": 140, "Greenville": 160, "Hill Valley": 150, "Shelbyville": 110}
    base = np.array([loc_base[x] for x in loc])
    per_sqft = np.array([rate[x] for x in loc])
    age_adj = (years - 1955) * 450
    price = base + area * per_sqft + bedrooms * 18000 + bathrooms * 12000 + age_adj
    noise = rng.normal(0, 30000, size=n)
    price = np.maximum(50000, price + noise)
    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location": loc,
        "year_built": years,
        "price": price.astype(float)
    })
    mask = rng.random(n) < 0.05
    df.loc[mask, "bathrooms"] = np.nan
    mask = rng.random(n) < 0.04
    df.loc[mask, "bedrooms"] = np.nan
    mask = rng.random(n) < 0.03
    df.loc[mask, "location"] = np.nan
    return df

def ensure_raw():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        df = generate_synthetic()
        df.to_csv(RAW_PATH, index=False)
        return
    df = pd.read_csv(RAW_PATH)
    if len(df) < 100:
        df = generate_synthetic()
        df.to_csv(RAW_PATH, index=False)

def clean_save():
    df = pd.read_csv(RAW_PATH)
    df = df.drop_duplicates()
    for col in ["area", "bedrooms", "bathrooms", "year_built"]:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
    if df["location"].isna().any():
        mode = df["location"].mode().iloc[0]
        df["location"] = df["location"].fillna(mode)
    df["bedrooms"] = df["bedrooms"].round().astype(int)
    df["bathrooms"] = df["bathrooms"].round().astype(int)
    df["area"] = df["area"].astype(float)
    df["year_built"] = df["year_built"].astype(int)
    df.to_csv(CLEAN_PATH, index=False)

if __name__ == "__main__":
    ensure_raw()
    clean_save()
