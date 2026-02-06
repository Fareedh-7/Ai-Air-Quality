import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def save_plot_no2_by_city(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.sort_values("NO2_Avg", ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["City"], plot_df["NO2_Avg"], color="#2C7FB8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("NO2 Avg")
    plt.title("Top Cities by NO2 Average")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_pollutant_means(df: pd.DataFrame, out_path: Path) -> None:
    pollutants = ["NO2_Avg", "PM25", "PM10", "SO2", "CO", "O3"]
    means = df[pollutants].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values, color="#1B9E77")
    plt.ylabel("Mean Level")
    plt.title("Average Pollutant Levels")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Air pollution AI pipeline")
    parser.add_argument("--data", default="data/datasets.csv", help="Path to CSV")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--model", default="models/no2_model.pkl", help="Model path")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    model_path = Path(args.model)

    outdir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = add_synthetic_pollutants(df)
    df = add_time_features(df)

    df.to_csv(outdir / "pollution_summary.csv", index=False)

    X = build_features(df)
    y = df["NO2_Avg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, model_path)

    df_predictions = df.copy()
    df_predictions["NO2_Predicted"] = model.predict(X)
    df_predictions.to_csv(outdir / "no2_predictions.csv", index=False)

    save_plot_no2_by_city(df, outdir / "no2_by_city.png")
    save_plot_pollutant_means(df, outdir / "pollutant_means.png")

    print("Saved:")
    print(f"- {outdir / 'pollution_summary.csv'}")
    print(f"- {outdir / 'no2_predictions.csv'}")
    print(f"- {outdir / 'metrics.json'}")
    print(f"- {outdir / 'no2_by_city.png'}")
    print(f"- {outdir / 'pollutant_means.png'}")
    print(f"- {model_path}")


if __name__ == "__main__":
    main()
