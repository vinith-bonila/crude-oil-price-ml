# Crude Oil Price Prediction using Machine Learning (CTRM Use Case)

This project builds a machine learning model to **forecast daily Brent crude oil prices** using historical time-series data.  
The goal is to demonstrate how ML-based forecasting supports **Commodity Trading & Risk Management (CTRM)** workflows such as:

- Price risk estimation  
- Hedging and exposure analysis  
- Trading decision support  
- Market scenario modelling  

---

## 📁 Dataset

- Source: Public Brent Crude daily price dataset (Kaggle)
- File: `data/brent_oil_prices.csv`
- Columns used: `Date`, `Price`

---

## 🧠 Approach

1. **Load & preprocess data**  
   - Parse dates  
   - Sort chronologically  
   - Handle missing values  

2. **Feature Engineering**  
   - Lag features: 1, 3, 7, 14 days  
   - Rolling statistics: 7-day & 30-day moving mean & std (volatility)

3. **Model Training**  
   - Linear Regression (baseline)  
   - Random Forest Regressor (main model)  

4. **Evaluation**  
   - MAE (Mean Absolute Error)  
   - RMSE (Root Mean Squared Error)  
   - Train-test split: 80/20 (time-based)

5. **Visualization**  
   - Actual vs Predicted Brent crude prices plotted with Matplotlib  

---

## 📈 Results

- Linear Regression: **MAE ≈ 1.07**, **RMSE ≈ 1.61**  
- Random Forest: **MAE ≈ 1.17**, **RMSE ≈ 1.74**  

The model captures short-term price trends well and demonstrates how ML can support trading & risk management decisions.

---

## 🔧 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python crude_oil_price_ml.py
