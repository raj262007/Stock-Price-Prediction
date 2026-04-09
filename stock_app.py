"""
Stock Price Predictor — Streamlit Frontend
Run: streamlit run stock_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Outfit:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
  .stApp { background: #060910; color: #f1f5f9; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0c1118 !important;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #0c1118;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #60a5fa !important; font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    transition: 0.2s !important;
    width: 100%;
  }
  .stButton > button:hover { background: #60a5fa !important; transform: translateY(-1px); }

  /* Selectbox, slider */
  .stSelectbox > div > div { background: #0c1118 !important; border-color: rgba(255,255,255,0.1) !important; color: #f1f5f9 !important; border-radius: 10px !important; }
  .stSlider > div > div > div { background: #3b82f6 !important; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0a1a2e, #0c1118);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 32px;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.03em; margin: 0;
    background: linear-gradient(90deg, #60a5fa, #4ade80);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-sub { color: #64748b; font-size: 1rem; margin-top: 8px; }

  /* Section heading */
  .sec-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 800;
    color: #60a5fa; margin: 28px 0 12px;
    display: flex; align-items: center; gap: 10px;
  }

  /* Info cards */
  .info-card {
    background: #0c1118;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 20px 22px; margin-bottom: 14px;
  }
  .info-card-title { font-weight: 600; font-size: 0.9rem; color: #60a5fa; margin-bottom: 4px; }
  .info-card-body  { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }

  /* Code box */
  .code-box {
    background: #070b10;
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 12px; padding: 16px 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem; color: #4ade80;
    margin-top: 8px;
  }

  /* Forecast table */
  .forecast-table { width: 100%; border-collapse: collapse; }
  .forecast-table th {
    background: #111820; color: #64748b;
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
  }
  .forecast-table td {
    padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.88rem; color: #f1f5f9;
  }
  .forecast-table tr:hover td { background: rgba(59,130,246,0.04); }
  .up   { color: #4ade80; font-weight: 600; }
  .down { color: #f87171; font-weight: 600; }

  /* Tag pill */
  .pill {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    color: #60a5fa; border: 1px solid rgba(59,130,246,0.25);
    border-radius: 999px; font-size: 0.72rem;
    padding: 3px 12px; margin: 3px 4px 3px 0;
  }

  div[data-testid="stMarkdownContainer"] h1,
  div[data-testid="stMarkdownContainer"] h2,
  div[data-testid="stMarkdownContainer"] h3 { color: #f1f5f9 !important; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CORE LOGIC (same as stock_predictor.py)
# ══════════════════════════════════════════════════════

def generate_stock_data(ticker, days, start_price, seed=42):
    np.random.seed(seed)
    dates = []
    current = datetime(2022, 1, 3)
    count = 0
    while count < days:
        if current.weekday() < 5:
            dates.append(current)
            count += 1
        current += timedelta(days=1)
    prices = [start_price]
    for i in range(1, days):
        trend     = 0.0003
        vol       = np.random.normal(0, 0.015)
        momentum  = 0.1 * (prices[-1] - prices[max(0,i-5)]) / prices[max(0,i-5)]
        mean_rev  = -0.05 * (prices[-1] - start_price) / start_price
        shock     = np.random.choice([0, np.random.normal(0, 0.04)], p=[0.97, 0.03])
        prices.append(max(prices[-1] * (1 + trend + vol + momentum + mean_rev + shock), 1.0))
    volume = np.random.randint(20_000_000, 80_000_000, size=days)
    df = pd.DataFrame({'Date': dates, 'Open': [p*np.random.uniform(0.995,1.005) for p in prices],
        'High': [p*np.random.uniform(1.005,1.025) for p in prices],
        'Low':  [p*np.random.uniform(0.975,0.995) for p in prices],
        'Close': prices, 'Volume': volume})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def add_features(df):
    df = df.copy()
    df['MA_5']  = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Momentum_5']  = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Volatility_10'] = df['Close'].rolling(10).std()
    df['Volatility_20'] = df['Close'].rolling(20).std()
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['BB_upper'] = df['MA_20'] + 2 * df['Volatility_20']
    df['BB_lower'] = df['MA_20'] - 2 * df['Volatility_20']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA_20']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

FEATURE_COLS = ['MA_5','MA_10','MA_20','MA_50','Momentum_5','Momentum_10',
                'Volatility_10','Volatility_20','RSI','BB_width','Volume_Change']

def train_model(df):
    X, y = df[FEATURE_COLS].values, df['Target'].values
    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    dates_test  = df.index[split:]
    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = LinearRegression()
    model.fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)
    return model, scaler, X_te_sc, y_tr, y_te, y_pred, dates_test, split

def forecast_future(model, scaler, df, days=30):
    current = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    last_price = df['Close'].iloc[-1]
    last_date  = df.index[-1]
    future_dates, future_prices = [], [last_price]
    for i in range(days):
        pred = model.predict(scaler.transform(current))[0]
        future_prices.append(pred)
        next_date = last_date + timedelta(days=i+1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        future_dates.append(next_date)
        current[0][0] = pred
    return future_dates, future_prices

# ── Chart helpers ─────────────────────────────────────
def dark_fig(figsize=(12,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0c1118')
    ax.set_facecolor('#060910')
    ax.tick_params(colors='#64748b', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, ha='right')
    return fig, ax

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    ticker = st.selectbox("Stock Ticker", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], index=0)
    days   = st.slider("Trading Days", 200, 800, 500, 50)
    start_price = st.slider("Start Price ($)", 50, 500, 150, 10)
    forecast_days = st.slider("Forecast Days", 7, 60, 30, 7)

    st.markdown("---")
    st.markdown("### 📐 Chart Options")
    show_ma    = st.checkbox("Show Moving Averages", True)
    show_bb    = st.checkbox("Show Bollinger Bands", True)
    show_rsi   = st.checkbox("Show RSI Chart",       True)
    show_vol   = st.checkbox("Show Volume",           False)

    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#475569;line-height:1.7'>
    <b style='color:#60a5fa'>Techniques Used:</b><br/>
    • Linear Regression<br/>
    • MinMax Scaling<br/>
    • Feature Engineering<br/>
    • RSI · Bollinger Bands<br/>
    • Moving Averages<br/>
    • Time-series Split
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">📈 Stock Price Predictor</p>
  <p class="hero-sub">AI Internship Project · Linear Regression · Technical Indicators · 30-Day Forecast</p>
  <div style="margin-top:14px">
    <span class="pill">Linear Regression</span>
    <span class="pill">RSI</span>
    <span class="pill">Bollinger Bands</span>
    <span class="pill">Moving Averages</span>
    <span class="pill">Momentum</span>
    <span class="pill">Scikit-Learn</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Run on button or first load
if run_btn or 'stock_ran' not in st.session_state:
    st.session_state['stock_ran'] = True

    with st.spinner("Generating data & training model..."):
        df_raw = generate_stock_data(ticker, days, start_price)
        df     = add_features(df_raw)
        model, scaler, X_te_sc, y_train, y_test, y_pred, dates_test, split = train_model(df)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        future_dates, future_prices = forecast_future(model, scaler, df, forecast_days)

    # ── Metrics Row ───────────────────────────────────
    st.markdown('<div class="sec-head">📊 Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",       f"${mae:.2f}",  help="Mean Absolute Error — avg dollar error per prediction")
    c2.metric("RMSE",      f"${rmse:.2f}", help="Root Mean Squared Error")
    c3.metric("MAPE",      f"{mape:.2f}%", help="Mean Absolute Percentage Error")
    c4.metric("R² Score",  f"{r2:.4f}",   delta=f"{r2*100:.1f}% variance explained")

    st.markdown("---")

    # ── Chart 1: Price + MAs ──────────────────────────
    st.markdown('<div class="sec-head">📉 Historical Price Chart</div>', unsafe_allow_html=True)
    fig1, ax1 = dark_fig((13, 4))
    ax1.plot(df.index, df['Close'], color='#60a5fa', lw=1.4, label='Close Price', zorder=3)
    if show_ma:
        ax1.plot(df.index, df['MA_20'], color='#f59e0b', lw=0.9, ls='--', label='MA 20', alpha=0.8)
        ax1.plot(df.index, df['MA_50'], color='#4ade80', lw=0.9, ls='--', label='MA 50', alpha=0.8)
    if show_bb:
        ax1.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.07, color='#60a5fa', label='Bollinger Bands')
    ax1.set_ylabel('Price (USD)', color='#94a3b8', fontsize=9)
    ax1.legend(facecolor='#0c1118', labelcolor='#94a3b8', fontsize=8, loc='upper left')
    ax1.grid(True, color='#1e293b', linewidth=0.4, alpha=0.6)
    ax1.yaxis.label.set_color('#94a3b8')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # ── Chart 2: Actual vs Predicted ──────────────────
    st.markdown('<div class="sec-head">🎯 Actual vs Predicted (Test Set)</div>', unsafe_allow_html=True)
    fig2, ax2 = dark_fig((13, 4))
    ax2.plot(dates_test, y_test, color='#60a5fa', lw=1.8, label='Actual Price')
    ax2.plot(dates_test, y_pred, color='#f87171', lw=1.5, ls='--', label='Predicted Price')
    ax2.fill_between(dates_test, y_test, y_pred, alpha=0.12, color='#a855f7')
    ax2.set_title(f'Test Period  |  MAE: ${mae:.2f}  |  R²: {r2:.4f}',
                  color='#64748b', fontsize=9, pad=8)
    ax2.set_ylabel('Price (USD)', color='#94a3b8', fontsize=9)
    ax2.legend(facecolor='#0c1118', labelcolor='#94a3b8', fontsize=8)
    ax2.grid(True, color='#1e293b', linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    col_a, col_b = st.columns(2)

    # ── Chart 3: RSI ──────────────────────────────────
    if show_rsi:
        with col_a:
            st.markdown('<div class="sec-head">⚡ RSI Indicator</div>', unsafe_allow_html=True)
            fig3, ax3 = dark_fig((7, 3))
            ax3.plot(df.index, df['RSI'], color='#a855f7', lw=1.0)
            ax3.axhline(70, color='#f87171', lw=0.8, ls='--', label='Overbought (70)')
            ax3.axhline(30, color='#4ade80', lw=0.8, ls='--', label='Oversold (30)')
            ax3.fill_between(df.index, df['RSI'], 70, where=(df['RSI']>=70), alpha=0.2, color='#f87171')
            ax3.fill_between(df.index, df['RSI'], 30, where=(df['RSI']<=30), alpha=0.2, color='#4ade80')
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('RSI', color='#94a3b8', fontsize=9)
            ax3.legend(facecolor='#0c1118', labelcolor='#94a3b8', fontsize=8)
            ax3.grid(True, color='#1e293b', linewidth=0.4, alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    # ── Volume ────────────────────────────────────────
    if show_vol:
        with col_b:
            st.markdown('<div class="sec-head">📦 Volume</div>', unsafe_allow_html=True)
            fig4, ax4 = dark_fig((7, 3))
            colors_v = ['#4ade80' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else '#f87171'
                        for i in range(len(df))]
            ax4.bar(df.index, df['Volume'], color=colors_v, alpha=0.7, width=0.8)
            ax4.set_ylabel('Volume', color='#94a3b8', fontsize=9)
            ax4.grid(True, color='#1e293b', linewidth=0.4, alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

    # ── Forecast Section ──────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="sec-head">🔮 {forecast_days}-Day Price Forecast</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        fig5, ax5 = dark_fig((8, 4))
        # Show last 60 days of actual + forecast
        hist_tail = df['Close'].iloc[-60:]
        ax5.plot(hist_tail.index, hist_tail.values, color='#60a5fa', lw=1.5, label='Historical')
        ax5.plot([hist_tail.index[-1]] + future_dates,
                 [hist_tail.values[-1]] + future_prices[1:],
                 color='#f59e0b', lw=1.8, ls='--', label=f'{forecast_days}-Day Forecast')
        ax5.fill_between([hist_tail.index[-1]] + future_dates,
                         [hist_tail.values[-1]] + [p*0.97 for p in future_prices[1:]],
                         [hist_tail.values[-1]] + [p*1.03 for p in future_prices[1:]],
                         alpha=0.1, color='#f59e0b')
        ax5.axvline(x=hist_tail.index[-1], color='#475569', lw=0.8, ls=':')
        ax5.set_ylabel('Price (USD)', color='#94a3b8', fontsize=9)
        ax5.legend(facecolor='#0c1118', labelcolor='#94a3b8', fontsize=8)
        ax5.grid(True, color='#1e293b', linewidth=0.4, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with col_f2:
        last_p = df['Close'].iloc[-1]
        final_p = future_prices[-1]
        change  = ((final_p - last_p) / last_p) * 100
        direction = "📈 UP" if change > 0 else "📉 DOWN"
        color_dir = "#4ade80" if change > 0 else "#f87171"

        st.markdown(f"""
        <div class="info-card">
          <div class="info-card-title">Last Actual Price</div>
          <div style="font-size:1.4rem;font-weight:700;color:#60a5fa">${last_p:.2f}</div>
        </div>
        <div class="info-card">
          <div class="info-card-title">{forecast_days}-Day Predicted</div>
          <div style="font-size:1.4rem;font-weight:700;color:#f59e0b">${final_p:.2f}</div>
        </div>
        <div class="info-card">
          <div class="info-card-title">Expected Change</div>
          <div style="font-size:1.2rem;font-weight:700;color:{color_dir}">{direction} {abs(change):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Forecast table (first 10 days)
        st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-top:12px;font-weight:600">First 10 Days:</div>', unsafe_allow_html=True)
        rows = ""
        for i, (d, p) in enumerate(zip(future_dates[:10], future_prices[1:11])):
            chg = ((p - last_p) / last_p) * 100
            cls = "up" if chg > 0 else "down"
            sym = "▲" if chg > 0 else "▼"
            rows += f"<tr><td>{d.strftime('%d %b')}</td><td>${p:.2f}</td><td class='{cls}'>{sym}{abs(chg):.1f}%</td></tr>"
        st.markdown(f"""
        <table class="forecast-table">
          <thead><tr><th>Date</th><th>Price</th><th>Change</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

    # ── Dataset Preview ───────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-head">🗃️ Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df[['Open','High','Low','Close','Volume','MA_20','RSI']].tail(10).round(2),
        use_container_width=True,
        hide_index=False
    )

    # ── Feature importance (Linear weights) ──────────
    st.markdown("---")
    st.markdown('<div class="sec-head">⚖️ Feature Weights (Linear Regression Coefficients)</div>', unsafe_allow_html=True)
    coeffs = pd.DataFrame({'Feature': FEATURE_COLS, 'Weight': model.coef_}).sort_values('Weight', ascending=False)
    fig6, ax6 = dark_fig((10, 4))
    colors_c = ['#4ade80' if w > 0 else '#f87171' for w in coeffs['Weight']]
    ax6.barh(coeffs['Feature'], coeffs['Weight'], color=colors_c, alpha=0.8, edgecolor='none')
    ax6.axvline(0, color='#475569', lw=0.8)
    ax6.set_xlabel('Coefficient Value', color='#94a3b8', fontsize=9)
    ax6.grid(True, axis='x', color='#1e293b', linewidth=0.4)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

else:
    st.info("👈 Configure settings in the sidebar and click **Run Prediction** to start.")
