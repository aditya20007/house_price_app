import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "clean.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "house_model.pkl")

def load_data() -> pd.DataFrame:
    if os.path.exists(CLEAN_PATH):
        return pd.read_csv(CLEAN_PATH)
    return pd.read_csv(RAW_PATH)

def build_preprocessor():
    numeric_features = ["area", "bedrooms", "bathrooms", "year_built"]
    categorical_features = ["location"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X = df[["area", "bedrooms", "bathrooms", "location", "year_built"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor()
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42, max_depth=None, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.0005, random_state=42, max_iter=2000)
    }
    best_name = None
    best_score = -1e9
    best_pipe = None
    for name, est in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", est)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"{name} R2: {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe
    print(f"Best model: {best_name} with R2 {best_score:.4f}")
    joblib.dump(best_pipe, MODEL_PATH)

if __name__ == "__main__":
    main()
