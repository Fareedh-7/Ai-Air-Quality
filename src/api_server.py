from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


DATA_PATH = Path("data/datasets.csv")
MODEL_PATH = Path("models/no2_model.pkl")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def add_synthetic_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NO2_Avg"] = (df["NO2_Low"] + df["NO2_High"]) / 2.0

    base = df["NO2_Avg"] * 1e5
    temp = df["Temperature"]
    humidity = df["Humidity"]
    wind = df["WindSpeed_m_s"]

    df["PM25"] = (base * 0.6 + humidity * 0.2 - wind * 1.5 + 5).clip(lower=0)
    df["PM10"] = (base * 0.9 + humidity * 0.3 - wind * 1.0 + 10).clip(lower=0)
    df["SO2"] = (base * 0.2 + (temp - 15) * 0.3 - wind * 0.5 + 2).clip(lower=0)
    df["CO"] = (base * 0.05 + humidity * 0.05 + 0.5).clip(lower=0)
    df["O3"] = (base * 0.3 + (temp - 10) * 0.8 - humidity * 0.1 + 3).clip(lower=0)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Latitude",
        "Longitude",
        "Temperature",
        "Humidity",
        "WindSpeed_m_s",
        "Year",
        "Month",
        "Day",
        "DayOfYear",
    ]
    return df[feature_cols]


app = FastAPI(title="Air Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Expected data/datasets.csv")

    df = pd.read_csv(DATA_PATH)
    df = add_synthetic_pollutants(df)
    df = add_time_features(df)
    df["City_Normalized"] = df["City"].str.strip().str.lower()
    return df


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@app.on_event("startup")
def startup_event() -> None:
    app.state.df = load_dataset()
    app.state.model = load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/cities")
def list_cities() -> dict:
    df = app.state.df
    cities = sorted(df["City"].dropna().unique().tolist())
    return {"cities": cities}


@app.get("/predict")
def predict(city: str) -> dict:
    if not city:
        raise HTTPException(status_code=400, detail="City is required")

    df = app.state.df
    city_norm = city.strip().lower()
    df_city = df[df["City_Normalized"] == city_norm]

    if df_city.empty:
        raise HTTPException(status_code=404, detail="City not found")

    row = df_city.sort_values("Date").iloc[-1]
    model = app.state.model

    if model is None:
        model = load_model()
        if model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not found. Run src/run_pipeline.py first.",
            )
        app.state.model = model

    features = build_features(pd.DataFrame([row]))
    no2_pred = float(model.predict(features)[0])

    return {
        "city": row["City"],
        "date": row["Date"].strftime("%Y-%m-%d"),
        "no2_avg": float(row["NO2_Avg"]),
        "no2_predicted": no2_pred,
        "pm25": float(row["PM25"]),
        "pm10": float(row["PM10"]),
        "so2": float(row["SO2"]),
        "co": float(row["CO"]),
        "o3": float(row["O3"]),
        "temperature": float(row["Temperature"]),
        "humidity": float(row["Humidity"]),
        "wind_speed": float(row["WindSpeed_m_s"]),
        "latitude": float(row["Latitude"]),
        "longitude": float(row["Longitude"]),
    }
