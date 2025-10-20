# train.py
import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os
import glob

def clean_original_price(df):
    # Convert empty strings or spaces to NaN
    df["original_price"].replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")

    # Impute missing values
    df["original_price"] = df.groupby(["make", "model"])["original_price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["original_price"] = df["original_price"].fillna(df["original_price"].median())
    return df

def load_and_merge_csvs():
    csv_files = glob.glob("*.csv")
    print(f"📂 Found CSV files: {csv_files}")

    if not csv_files:
        raise FileNotFoundError("❌ No CSV files found in repository!")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            print(f"✅ Loaded {f} with {len(df)} rows.")
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {f} due to error: {e}")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.drop_duplicates(inplace=True)
    print(f"📊 Combined dataset has {len(merged_df)} rows after merging.")
    return merged_df

def main():
    print("🚀 Starting training pipeline...")

    df = load_and_merge_csvs()

    # Columns to use
    features = ["car_name", "yr_mfr", "kms_run", "fuel_type", "city", 
                "total_owners", "body_type", "transmission", 
                "make", "model", "original_price"]
    target = "sale_price"

    # Keep only required columns
    df = df[[c for c in features + [target] if c in df.columns]]

    # Clean price column
    df = clean_original_price(df)

    # Handle missing general values
    df = df.fillna("missing")

    # Convert numeric columns
    df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce")

    # Feature engineering
    df["car_age"] = 2025 - df["yr_mfr"]
    df.drop(columns=["yr_mfr"], inplace=True)

    # Encode categorical
    cat_cols = ["car_name", "fuel_type", "city", "total_owners", 
                "body_type", "transmission", "make", "model"]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Split data
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log-transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Model
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("🧠 Training model...")
    model.fit(X_train, y_train_log)

    # Predict
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"✅ RMSE: ₹{rmse:,.0f}")

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(encoder, "artifacts/encoder.pkl")

    print("💾 Model and encoder saved to /artifacts")

if __name__ == "__main__":
    main()
