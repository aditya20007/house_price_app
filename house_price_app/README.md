# House Price Prediction App

This project is a complete house price prediction system built with Python, Scikit-Learn, and Flask, including a responsive HTML/CSS frontend and a trained machine learning model.

## Tech Stack
- Python, NumPy, Pandas
- Scikit-Learn
- Flask
- HTML/CSS
- Gunicorn (deployment)

## Folder Structure
```
house_price_app/
├── data/
│   └── raw.csv
├── model/
│   └── house_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── requirements.txt
├── README.md
├── Procfile
└── .gitignore
```

## How to Run
1. Create a virtual environment and install dependencies:
```
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r house_price_app/requirements.txt
```

2. Generate and clean data:
```
python house_price_app/src/preprocess.py
```

3. Train the model:
```
python house_price_app/src/train.py
```

4. Start the Flask app for local testing:
```
python house_price_app/app/app.py
```

## API Endpoints
- `GET /` renders the prediction form.
- `POST /predict` accepts form or JSON with:
  - `area` (float), `bedrooms` (int), `bathrooms` (int), `location` (string), `year_built` (int)
  - Returns predicted price in JSON or renders the page with the result.

## Screenshots
- Add screenshots of the UI and sample predictions.

## Future Improvements
- Persist training metrics and add model versioning.
- Add more features (garage, lot size, school ratings).
- Integrate a real dataset and geospatial features.
- Containerize with Docker for deployment.
