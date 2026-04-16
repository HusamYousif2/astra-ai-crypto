# market/nlp_engine.py

import re

import feedparser  # type: ignore
from transformers import pipeline  # type: ignore

# Initialize FinBERT once at startup
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Major crypto news RSS feeds
RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cryptoslate": "https://cryptoslate.com/feed/",
}


def clean_text(text):
    """
    Remove HTML tags and extra spaces from RSS content.
    """
    clean = re.compile(r"<.*?>")
    text_without_html = re.sub(clean, "", text)
    return " ".join(text_without_html.split())


def fetch_latest_news():
    """
    Fetch the latest articles from all configured RSS feeds.
    """
    all_articles = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:5]:
                all_articles.append(
                    {
                        "source": source,
                        "title": getattr(entry, "title", ""),
                        "summary": clean_text(getattr(entry, "summary", "")),
                        "link": getattr(entry, "link", ""),
                    }
                )
        except Exception as e:
            print(f"Failed to fetch from {source}: {e}")

    return all_articles


def get_coin_keywords(coin_symbol):
    """
    Build a keyword list for a coin symbol and its common name.
    """
    coin_symbol = coin_symbol.lower()

    name_map = {
        "btc": ["btc", "bitcoin"],
        "eth": ["eth", "ethereum"],
        "ada": ["ada", "cardano"],
        "doge": ["doge", "dogecoin"],
        "sol": ["sol", "solana"],
        "xrp": ["xrp", "ripple"],
        "bnb": ["bnb", "binance coin", "binance"],
        "dot": ["dot", "polkadot"],
        "ltc": ["ltc", "litecoin"],
    }

    return name_map.get(coin_symbol, [coin_symbol])


def classify_sentiment_label(score):
    """
    Convert a numeric score into a business-friendly label.
    """
    if score > 0.20:
        return "BULLISH"
    if score < -0.20:
        return "BEARISH"
    return "NEUTRAL"


def analyze_article_sentiment(article):
    """
    Analyze sentiment for a single article using FinBERT.
    """
    text_for_ai = f"{article['title']}. {article['summary']}".strip()
    text_for_ai = text_for_ai[:500]

    result = sentiment_analyzer(text_for_ai)[0]
    raw_label = result["label"].lower()
    confidence = float(result["score"])

    if raw_label == "positive":
        signed_score = confidence
        final_label = "BULLISH"
    elif raw_label == "negative":
        signed_score = -confidence
        final_label = "BEARISH"
    else:
        signed_score = 0.0
        final_label = "NEUTRAL"

    return {
        "sentiment_label": final_label,
        "sentiment_score": round(signed_score, 3),
        "model_confidence": round(confidence, 3),
    }


def build_news_summary(coin_symbol, overall_label, article_count, top_articles):
    """
    Build a professional summary for the news panel.
    """
    if article_count == 0:
        return f"No meaningful recent news was detected for {coin_symbol.upper()}."

    if overall_label == "BULLISH":
        tone = "Recent news flow is supportive overall."
    elif overall_label == "BEARISH":
        tone = "Recent news flow is negative overall."
    else:
        tone = "Recent news flow is mixed and not strongly directional."

    top_titles = [article["title"] for article in top_articles[:2] if article.get("title")]

    if top_titles:
        return f"{tone} {article_count} relevant article(s) were detected. Key headlines include: " + " | ".join(top_titles)
    return f"{tone} {article_count} relevant article(s) were detected."


def analyze_coin_sentiment(coin_symbol):
    """
    Analyze recent news sentiment for a specific coin.
    Returns both aggregate output and article-level detail.
    """
    articles = fetch_latest_news()
    coin_keywords = get_coin_keywords(coin_symbol)

    relevant_articles = []
    for article in articles:
        text_to_search = f"{article['title']} {article['summary']}".lower()

        if any(keyword in text_to_search for keyword in coin_keywords):
            relevant_articles.append(article)

    if not relevant_articles:
        return {
            "score": 0.0,
            "label": "NEUTRAL",
            "article_count": 0,
            "summary": f"No meaningful recent news was detected for {coin_symbol.upper()}.",
            "articles": [],
        }

    enriched_articles = []
    total_score = 0.0

    for article in relevant_articles:
        sentiment_data = analyze_article_sentiment(article)

        enriched_article = {
            "source": article["source"],
            "title": article["title"],
            "summary": article["summary"],
            "link": article["link"],
            "sentiment_label": sentiment_data["sentiment_label"],
            "sentiment_score": sentiment_data["sentiment_score"],
            "model_confidence": sentiment_data["model_confidence"],
        }

        enriched_articles.append(enriched_article)
        total_score += sentiment_data["sentiment_score"]

    avg_score = total_score / len(enriched_articles)
    final_label = classify_sentiment_label(avg_score)

    # Sort articles by absolute sentiment impact
    enriched_articles = sorted(
        enriched_articles,
        key=lambda x: abs(x["sentiment_score"]),
        reverse=True,
    )

    summary = build_news_summary(
        coin_symbol=coin_symbol,
        overall_label=final_label,
        article_count=len(enriched_articles),
        top_articles=enriched_articles,
    )

    return {
        "score": round(avg_score, 3),
        "label": final_label,
        "article_count": len(enriched_articles),
        "summary": summary,
        "articles": enriched_articles[:5],
    }


if __name__ == "__main__":
    print("Fetching and analyzing news for BTC... Please wait.")
    result = analyze_coin_sentiment("BTC")
    print(result)