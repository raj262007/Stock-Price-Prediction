# 📈 Stock Price Predictor

> Predicting stock closing prices using Linear Regression and technical indicators.

## 🔍 Overview
This project builds a machine learning model that predicts the **next day's closing price** of a stock using historical price data and technical analysis features.

## 🧠 Techniques Used
| Component | Details |
|-----------|---------|
| Model | Linear Regression (scikit-learn) |
| Features | Moving Averages, RSI, Bollinger Bands, Momentum, Volatility |
| Scaling | MinMaxScaler |
| Evaluation | MAE, RMSE, MAPE, R² Score |
| Visualization | Matplotlib (dark theme charts) |

## 📁 Project Structure
```
stock_price_predictor/
│
├── stock_predictor.py         # Main script (data → features → train → predict → plot)
├── requirements.txt           # Python dependencies
├── stock_prediction_results.png  # Output chart (auto-generated)
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the predictor
```bash
python stock_predictor.py
```

### 3. Output
- Console prints model metrics (MAE, RMSE, R², MAPE)
- Console prints 30-day price forecast
- `stock_prediction_results.png` is saved automatically

## 📊 Features Engineered
- **MA 5 / 10 / 20 / 50** — Short & long-term moving averages
- **Momentum** — Price change over 5 and 10 days
- **Volatility** — Rolling standard deviation (10 & 20 day)
- **RSI** — Relative Strength Index (overbought/oversold signal)
- **Bollinger Bands** — Price channel width indicator
- **Volume Change** — Daily volume percentage change

## 🔧 To Use Real Data (Optional Upgrade)
```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
```
Replace the `generate_stock_data()` call with this one line.

## 📈 Sample Output
```
MODEL EVALUATION — AAPL Stock Predictor
═══════════════════════════════════════
MAE  (Mean Absolute Error)     : $2.14
RMSE (Root Mean Squared Error) : $2.87
MAPE (Mean Abs % Error)        : 1.42%
R²   Score                     : 0.9823  (98.2% variance explained)
```

## 🛠️ Tech Stack
- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- Matplotlib

---
*Internship Project — ML Fundamentals*
