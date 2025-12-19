import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import sys
import os

# ---------------- PATH SETUP ----------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agent import get_agent_response, finance_glossary
from backend.sentiment import analyze_sentiment

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MarketVision Pro",
    page_icon="📈",
    layout="wide"
)

# ---------------- HTML FRONTEND ----------------
def render_html_ui():
    html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MarketVision Pro</title>

<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
.glass {
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.15);
}
</style>
</head>

<body class="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-violet-900 text-slate-100">
<div class="max-w-7xl mx-auto p-6">

<header class="mb-6">
  <h1 class="text-3xl font-bold">📈 MarketVision Pro</h1>
  <p class="text-slate-300">Live watchlist + AI forecasting + Market Bot</p>
</header>

<section class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
  <div class="glass rounded-xl p-4">
    <p class="text-sm text-slate-300">Watchlist</p>
    <p class="text-xl font-bold mt-2">AAPL, MSFT, AMZN, GOOGL</p>
  </div>

  <div class="glass rounded-xl p-4">
    <p class="text-sm text-slate-300">Forecast Engine</p>
    <p class="mt-2 text-green-400 font-bold">Prophet + ML Ready</p>
  </div>

  <div class="glass rounded-xl p-4">
    <p class="text-sm text-slate-300">AI Market Bot</p>
    <p class="mt-2 text-indigo-400 font-bold">Agentic AI Enabled</p>
  </div>
</section>

<section class="glass rounded-xl p-4">
  <h2 class="text-lg font-semibold mb-2">Sample Market Trend</h2>
  <canvas id="chart"></canvas>
</section>

</div>

<script>
const ctx = document.getElementById("chart");
new Chart(ctx, {
  type: "line",
  data: {
    labels: ["Mon","Tue","Wed","Thu","Fri"],
    datasets: [{
      label: "Demo Price",
      data: [120,130,125,140,150],
      borderColor: "#22d3ee",
      tension: 0.4
    }]
  }
});
</script>

</body>
</html>
"""
    components.html(html_code, height=850, scrolling=True)

# ---------------- STREAMLIT BACKEND ----------------
def main():

    # Render HTML UI
    render_html_ui()

    st.divider()
    st.subheader("📊 Live Market Analysis (Backend)")

    ticker = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA"]
    )

    days = st.slider("Forecast Days", 1, 30, 7)

    if ticker:
        df = yf.download(ticker, period="1y", interval="1d")
        df = df.reset_index()
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna()

        if len(df) > 30:
            model = Prophet(daily_seasonality=True)
            model.fit(df)

            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["ds"], y=df["y"],
                name="History",
                line=dict(color="#6366f1")
            ))

            future_df = forecast.tail(days)
            fig.add_trace(go.Scatter(
                x=future_df["ds"], y=future_df["yhat"],
                name="Forecast",
                line=dict(color="#22d3ee", dash="dash")
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI CHATBOT ----------------
    st.divider()
    st.subheader("🤖 Market Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about markets, stocks, or news"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                reply = get_agent_response(prompt)

                try:
                    sentiment = analyze_sentiment(prompt)
                    label = sentiment.get("FinBERT Label", "Neutral")
                    st.markdown(reply)
                    st.caption(f"Sentiment: {label}")
                except:
                    st.markdown(reply)

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )

    # ---------------- GLOSSARY ----------------
    st.divider()
    st.subheader("📚 Financial Glossary")

    term = st.text_input("Enter term (e.g., EBITDA)")
    if st.button("Search"):
        res = finance_glossary.get(term.lower().strip())
        if res:
            st.success(res)
        else:
            st.error("Term not found")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
