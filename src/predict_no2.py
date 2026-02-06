import argparse
from pathlib import Path

import joblib
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NO2 average from features")
    parser.add_argument("--data", required=True, help="CSV with input rows")
    parser.add_argument("--model", default="models/no2_model.pkl", help="Model path")
    parser.add_argument("--out", default="results/no2_predictions_input.csv", help="Output CSV")
    args = parser.parse_args()

    model = joblib.load(Path(args.model))
    df = pd.read_csv(args.data)
    df = add_time_features(df)
    X = build_features(df)

    df_out = df.copy()
    df_out["NO2_Predicted"] = model.predict(X)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)

    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
