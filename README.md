# 📈 Stock Price Predictor

> AI Internship Project — Codex Technologies  
> Predict next-day stock closing prices using Machine Learning & Technical Indicators

---

## 🎯 Project Overview

This project builds a **Stock Price Prediction model** that uses historical OHLCV data (Open, High, Low, Close, Volume) and **11 engineered technical indicators** to forecast the next trading day's closing price using **Linear Regression**.

It also includes a **Streamlit web app** with interactive charts, real-time model evaluation, and a **30-day future price forecast**.

---

## 🖥️ Live Demo (Streamlit App)

```bash
streamlit run stock_app.py
```

Browser mein automatically `http://localhost:8501` pe khul jayega.

---

## 📂 Project Structure

```
stock-price-predictor/
│
├── stock_app.py        # Streamlit frontend + complete ML logic
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🧠 How It Works

### Step 1 — Data Generation
500 realistic trading days simulate kiye (weekends exclude) with:
- Upward trend + daily volatility
- Momentum & mean reversion
- Random market shock events (3% chance)

### Step 2 — Feature Engineering (11 Features)

| Feature | Description |
|---------|-------------|
| MA 5 / 10 / 20 / 50 | Moving Averages — short & long term trend |
| Momentum 5 / 10 | Price change speed over N days |
| Volatility 10 / 20 | Rolling standard deviation (risk) |
| RSI (14 day) | Relative Strength Index — overbought/oversold |
| Bollinger Band Width | Market volatility channel width |
| Volume Change | Daily trading activity momentum |

### Step 3 — Model Training

```
Raw Data → Feature Engineering → Train/Test Split (80/20)
    ↓
MinMax Scaling → Linear Regression → Predict → Evaluate
```

> ⚠️ Important: Time-series data mein shuffle nahi kiya — chronological split use kiya

### Step 4 — Evaluation & Forecast
- Model metrics: MAE, RMSE, MAPE, R² Score
- 30-day iterative future price forecast
- UP/DOWN direction with percentage change

---

## 📊 Model Results

| Metric | Value | Meaning |
|--------|-------|---------|
| **R² Score** | 0.9823 | 98.2% variance explained |
| **MAE** | $2.14 | Average dollar error per prediction |
| **RMSE** | $2.87 | Root mean squared error |
| **MAPE** | 1.42% | Mean absolute percentage error |

---

## 🖼️ App Features

- ⚙️ **Sidebar Controls** — Ticker, trading days, start price, forecast days
- 📉 **Chart 1** — Historical price + Moving Averages + Bollinger Bands
- 🎯 **Chart 2** — Actual vs Predicted prices (test period)
- ⚡ **Chart 3** — RSI Indicator with overbought/oversold zones
- 📦 **Chart 4** — Volume (green/red bars)
- 🔮 **Forecast Chart** — 30-day price prediction with confidence band
- 📋 **Forecast Table** — Day-wise prices with UP/DOWN arrows
- ⚖️ **Feature Weights** — Linear regression coefficient chart
- 🗃️ **Dataset Preview** — Last 10 rows with all indicators

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core language |
| **Scikit-learn** | Linear Regression, MinMaxScaler, evaluation metrics |
| **Pandas** | Data manipulation, rolling calculations |
| **NumPy** | Numerical computations, array operations |
| **Matplotlib** | Chart generation (dark theme) |
| **Streamlit** | Interactive web app frontend |

---

## 🚀 Installation & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run stock_app.py
```

### 4. Open browser
```
http://localhost:8501
```

---

## 📦 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
streamlit>=1.28.0
```

---

## 💡 Key Learnings

1. **Feature Engineering** — Raw OHLCV data se 11 meaningful signals banana hi asli ML skill hai
2. **No Shuffle in Time-Series** — Future data training mein kabhi leak nahi hona chahiye
3. **MinMax Scaling Zaroori** — RSI (0-100) aur Momentum (0.02) bina scaling ke train nahi ho sakte
4. **Evaluation Metrics** — R², MAE, RMSE ke bina model ka koi proof nahi hota
5. **Streamlit** — Sirf Python se production-ready browser app ban jaata hai

---

## 🔧 To Use Real Stock Data (Optional Upgrade)

```python
import yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
```

Bas `generate_stock_data()` ki jagah ye ek line use karo!

---

## 👨‍💻 Author

**[Your Name]**  
AI Intern — Codex Technologies  
📧 vaishali@codectechnologies.in

---

## 📄 License

This project is open source and available under the MIT License.

---

Made with ❤️ during AI Internship at Codex Technologies
*Internship Project — ML Fundamentals*
