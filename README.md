# 📈 Stock Price Predictor

> AI Internship Project — Codex Technologies
> Predicting next-day stock closing prices using Machine Learning & Technical Indicators

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Project Overview

This project is a **Stock Price Prediction System** built as part of my AI Internship at **Codex Technologies**. It uses historical stock market data (OHLCV — Open, High, Low, Close, Volume) combined with **11 engineered technical indicators** to predict the next trading day's closing price using **Linear Regression**.

The project also features a fully interactive **Streamlit web application** with live charts, model evaluation metrics, and a **30-day future price forecast**.

---

## 🖥️ Run the App

```bash
streamlit run stock_app.py
```

Opens automatically at `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
stock-price-predictor/
│
├── stock_app.py        ← Main file — Streamlit UI + complete ML logic
├── requirements.txt    ← Python dependencies
└── README.md           ← Project documentation
```

---

## 🧠 How It Works

### Step 1 — Data Generation
Generates 500 realistic trading days (weekends excluded) using:
- Long-term upward trend
- Daily random volatility (Normal distribution)
- Price momentum and mean reversion
- Rare market shock events (3% probability)

### Step 2 — Feature Engineering

The model doesn't use raw prices directly. Instead, **11 technical indicators** are computed:

| Feature | Formula | What It Signals |
|---------|---------|----------------|
| MA 5 / 10 / 20 / 50 | Rolling mean of Close over N days | Short & long-term trend direction |
| Momentum 5 / 10 | (Close today − Close N days ago) / Close N | Speed and direction of price change |
| Volatility 10 / 20 | Rolling standard deviation of Close | Daily risk and uncertainty |
| RSI (14-day) | 100 − 100 / (1 + avg gain / avg loss) | Overbought (>70) or Oversold (<30) |
| Bollinger Band Width | (Upper − Lower) / MA 20 | Volatility compression or expansion |
| Volume Change | (Volume today − Volume yesterday) / yesterday | Trading activity momentum |

### Step 3 — Model Pipeline

```
Raw OHLCV Data
      ↓
Feature Engineering (11 indicators)
      ↓
Train / Test Split — 80% / 20% (chronological, no shuffle)
      ↓
MinMax Scaling → [0, 1] range
      ↓
Linear Regression Training
      ↓
Predict → Evaluate → 30-Day Forecast
```

> ⚠️ **Note:** Data is never shuffled — time-series integrity is preserved to prevent future data leakage into training.

### Step 4 — Output
- Evaluation metrics printed to screen
- 3 dark-theme charts generated
- 30-day iterative price forecast with UP/DOWN direction

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **R² Score** | 0.9823 | Model explains 98.2% of price variance |
| **MAE** | $2.14 | Average dollar error per prediction |
| **RMSE** | $2.87 | Penalizes large errors more than MAE |
| **MAPE** | 1.42% | Only 1.42% average percentage error |

---

## 🖼️ App Features

| Feature | Description |
|---------|-------------|
| ⚙️ Sidebar Controls | Choose ticker, days, start price, forecast days |
| 📉 Price Chart | Historical close + MA20 + MA50 + Bollinger Bands |
| 🎯 Prediction Chart | Actual vs Predicted prices on the test set |
| ⚡ RSI Chart | Overbought / Oversold zones visualized |
| 📦 Volume Chart | Green/red volume bars per day |
| 🔮 Forecast Chart | 30-day future price prediction with confidence band |
| 📋 Forecast Table | Day-by-day prices with ▲ UP / ▼ DOWN arrows |
| ⚖️ Feature Weights | Bar chart of Linear Regression coefficients |
| 🗃️ Dataset Preview | Last 10 rows with all engineered indicators |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Core programming language |
| Scikit-learn | ≥0.24 | Linear Regression, MinMaxScaler, metrics |
| Pandas | ≥1.3 | Data manipulation and rolling calculations |
| NumPy | ≥1.21 | Numerical operations and array math |
| Matplotlib | ≥3.4 | Dark-theme chart generation |
| Streamlit | ≥1.28 | Interactive browser-based frontend |

---

## 🚀 Installation & Setup


### 1. Install all dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit app
```bash
streamlit run stock_app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 📦 requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
streamlit>=1.28.0
```

---

## 💡 Key Learnings

1. **Feature engineering matters more than the model** — 11 well-crafted indicators from raw OHLCV data gave 98.2% R² with simple Linear Regression.
2. **Never shuffle time-series data** — chronological split is essential to prevent future data leaking into training.
3. **Scaling is critical** — RSI ranges 0–100 while Momentum is ~0.02; MinMaxScaler normalizes both to [0,1] so neither dominates.
4. **Evaluation proves the model works** — without R², MAE, and RMSE, there is no evidence the model is actually learning anything.
5. **Streamlit removes the barrier** — a fully interactive ML web app in pure Python with no HTML, CSS, or JavaScript needed.

---

## 🔧 Optional Upgrade — Use Real Stock Data

Replace the `generate_stock_data()` call with one line using `yfinance`:

```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
```

Install yfinance:
```bash
pip install yfinance
```

---

## 👨‍💻 Author

**[Pawan Singh]**
AI Intern — Codex Technologies



---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!**

Made with ❤️ during AI Internship at **Codex Technologies**

</div>
