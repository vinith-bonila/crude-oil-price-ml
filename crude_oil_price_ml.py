import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. Load the dataset
# -----------------------------

DATA_PATH = os.path.join("data", "brent_oil_prices.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Please make sure 'brent_oil_prices.csv' is placed in the data/ folder."
    )

df = pd.read_csv(DATA_PATH)

# Try common column names; adjust if necessary
possible_date_cols = ["Date", "date", "DATE"]
possible_price_cols = ["Price", "price", "Close", "ClosePrice", "Value"]

date_col = None
price_col = None

for c in df.columns:
    if c in possible_date_cols:
        date_col = c
    if c in possible_price_cols:
        price_col = c

if date_col is None or price_col is None:
    raise ValueError(
        f"Could not auto-detect date/price columns. "
        f"Columns present: {list(df.columns)}. "
        f"Please rename the date column to 'Date' and the price column to 'Price' in the CSV "
        f"or modify 'possible_date_cols' and 'possible_price_cols' in the script."
    )

# Parse date and sort
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col).reset_index(drop=True)

df = df[[date_col, price_col]].copy()
df.columns = ["date", "price"]

print("Data loaded:")
print(df.head())

# -----------------------------
# 2. Feature Engineering
# -----------------------------

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    # Lag features: previous days' prices
    data["lag_1"] = data["price"].shift(1)
    data["lag_3"] = data["price"].shift(3)
    data["lag_7"] = data["price"].shift(7)
    data["lag_14"] = data["price"].shift(14)

    # Rolling mean and std (volatility)
    data["roll_mean_7"] = data["price"].rolling(window=7).mean()
    data["roll_std_7"] = data["price"].rolling(window=7).std()
    data["roll_mean_30"] = data["price"].rolling(window=30).mean()
    data["roll_std_30"] = data["price"].rolling(window=30).std()

    return data

df_feat = create_features(df)

# Drop rows with NaNs (due to lag/rolling)
df_feat = df_feat.dropna().reset_index(drop=True)

feature_cols = [
    "lag_1", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_std_7",
    "roll_mean_30", "roll_std_30"
]

X = df_feat[feature_cols].values
y = df_feat["price"].values
dates = df_feat["date"].values

print(f"\nTotal samples after feature engineering: {len(df_feat)}")

# -----------------------------
# 3. Train–Test Split (time-based)
# -----------------------------

split_index = int(len(df_feat) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = dates[:split_index], dates[split_index:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# -----------------------------
# 4. Baseline Model: Linear Regression
# -----------------------------

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lr = lin_reg.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5  # manual RMSE

print("\n=== Linear Regression Performance ===")
print(f"MAE:  {mae_lr:.3f}")
print(f"RMSE: {rmse_lr:.3f}")

# -----------------------------
# 5. Main Model: Random Forest Regressor
# -----------------------------

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5  # manual RMSE

print("\n=== Random Forest Performance ===")
print(f"MAE:  {mae_rf:.3f}")
print(f"RMSE: {rmse_rf:.3f}")


# -----------------------------
# 6. Plot: Actual vs Predicted
# -----------------------------

plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label="Actual Price")
plt.plot(dates_test, y_pred_rf, label="Predicted Price (Random Forest)")
plt.xlabel("Date")
plt.ylabel("Brent Crude Price")
plt.title("Actual vs Predicted Brent Crude Oil Prices")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Simple Text Summary
# -----------------------------

print("\nModel summary:")
print(" - Features: lagged prices + rolling mean/std over 7 and 30 days.")
print(" - Train/test split: time-based (80% / 20%).")
print(" - Models used: Linear Regression (baseline), Random Forest (main).")
print("This model can be used as a basic price-forecasting component in a CTRM context.")
