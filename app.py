import streamlit as st
import requests
import pandas as pd
import joblib
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NewsAPI key
NEWS_API_KEY = "697f3fe062df4e14aec7752db49f06e4"

# ----- News Functions -----
def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [a["title"] for a in articles if "title" in a][:10]

# ----- Twitter Functions -----
def fetch_tweets(ticker, count=10):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{ticker} since:2024-01-01').get_items()):
        if i >= count:
            break
        tweets.append(tweet.content)
    return tweets

# ----- Analysis Functions -----
def analyze_sentiment(texts):
    vectors = vectorizer.transform(texts)
    return model.predict(vectors)

def assess_risk(predictions):
    negative = sum(1 for p in predictions if p == 0)
    total = len(predictions)
    ratio = negative / total if total else 0
    if ratio > 0.6: return "🔴 High Risk"
    elif ratio > 0.3: return "🟠 Medium Risk"
    else: return "🟢 Low Risk"

# ----- Visualization -----
def plot_sentiments(sentiments, title):
    labels = ["Positive", "Negative"]
    counts = [sum(sentiments == 1), sum(sentiments == 0)]

    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=counts, ax=ax, palette="Set2")
    ax.set_title(title)
    st.pyplot(fig)

# ----- Streamlit App -----
st.set_page_config(page_title="Stock Sentiment & Risk Analyzer", layout="wide")
st.title("📊 Stock Sentiment & Risk Analyzer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")

if st.button("Analyze"):
    if ticker:
        st.subheader("📰 News Analysis")
        news = fetch_news(ticker)
        if news:
            news_preds = analyze_sentiment(news)
            df_news = pd.DataFrame({
                "News Headline": news,
                "Sentiment": ["Positive" if p == 1 else "Negative" for p in news_preds]
            })
            st.dataframe(df_news)
            plot_sentiments(news_preds, "News Sentiment Breakdown")
            st.success(f"🧮 Risk Level (News): {assess_risk(news_preds)}")
        else:
            st.warning("No news found.")

        st.subheader("🐦 Twitter Analysis")
        tweets = fetch_tweets(ticker)
        if tweets:
            tweet_preds = analyze_sentiment(tweets)
            df_tweets = pd.DataFrame({
                "Tweet": tweets,
                "Sentiment": ["Positive" if p == 1 else "Negative" for p in tweet_preds]
            })
            st.dataframe(df_tweets)
            plot_sentiments(tweet_preds, "Twitter Sentiment Breakdown")
            st.success(f"🧮 Risk Level (Twitter): {assess_risk(tweet_preds)}")
        else:
            st.warning("No tweets found.")

    else:
        st.warning("Please enter a valid stock ticker.")
