# market/nlp_engine.py

import feedparser
from transformers import pipeline
import re

# Initialize the FinBERT model. 
# The first time you run this, it will download the model (approx. 400MB).
# FinBERT is specifically trained on financial data.
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Major crypto news RSS feeds (Free and no rate limits)
RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cryptoslate": "https://cryptoslate.com/feed/"
}

def clean_text(text):
    """
    Remove HTML tags and extra spaces from the RSS summary.
    """
    clean = re.compile('<.*?>')
    text_without_html = re.sub(clean, '', text)
    return " ".join(text_without_html.split())

def fetch_latest_news():
    """
    Fetch the latest articles from all defined RSS feeds.
    Returns a list of dictionaries containing title, summary, and link.
    """
    all_articles = []
    
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            # Get top 5 recent articles from each source to avoid overloading the model
            for entry in feed.entries[:5]:
                all_articles.append({
                    "source": source,
                    "title": entry.title,
                    "summary": clean_text(getattr(entry, 'summary', '')),
                    "link": entry.link
                })
        except Exception as e:
            print(f"Failed to fetch from {source}: {e}")
            
    return all_articles

def analyze_coin_sentiment(coin_symbol):
    """
    Filter news for a specific coin and calculate the overall AI sentiment score.
    Returns a sentiment score between -1.0 (Highly Bearish) and 1.0 (Highly Bullish).
    """
    articles = fetch_latest_news()
    
    # Filter articles that mention the coin symbol or its common name
    # e.g., 'BTC' or 'Bitcoin'
    coin_keywords = [coin_symbol.lower()]
    
    # Add common names mapping
    name_map = {
        "btc": "bitcoin", "eth": "ethereum", "ada": "cardano", 
        "doge": "dogecoin", "sol": "solana", "xrp": "ripple"
    }
    if coin_symbol.lower() in name_map:
        coin_keywords.append(name_map[coin_symbol.lower()])
        
    relevant_articles = []
    for article in articles:
        text_to_search = (article["title"] + " " + article["summary"]).lower()
        if any(keyword in text_to_search for keyword in coin_keywords):
            relevant_articles.append(article)
            
    if not relevant_articles:
        return {"score": 0.0, "label": "NEUTRAL", "article_count": 0}
        
    total_score = 0
    
    # Pass text to FinBERT
    for article in relevant_articles:
        # We combine title and summary for better context
        text_for_ai = f"{article['title']}. {article['summary']}"
        
        # FinBERT has a token limit, so we truncate text to 512 tokens
        text_for_ai = text_for_ai[:500] 
        
        result = sentiment_analyzer(text_for_ai)[0]
        label = result['label']
        confidence = result['score']
        
        # Convert label to a mathematical weight
        if label == "positive":
            total_score += confidence
        elif label == "negative":
            total_score -= confidence
        # 'neutral' adds 0
        
    # Calculate average sentiment (-1 to 1)
    avg_score = total_score / len(relevant_articles)
    
    # Determine final label based on the average score
    if avg_score > 0.2:
        final_label = "BULLISH"
    elif avg_score < -0.2:
        final_label = "BEARISH"
    else:
        final_label = "NEUTRAL"
        
    return {
        "score": round(avg_score, 3),
        "label": final_label,
        "article_count": len(relevant_articles)
    }

# --- Simple block to test the code directly in the terminal ---
if __name__ == "__main__":
    print("Fetching and analyzing news for BTC... Please wait.")
    result = analyze_coin_sentiment("BTC")
    print(f"BTC Sentiment Result: {result}")