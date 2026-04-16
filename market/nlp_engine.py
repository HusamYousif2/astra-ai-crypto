import re
import logging

import feedparser  # type: ignore


logger = logging.getLogger(__name__)

sentiment_analyzer = None


RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cryptoslate": "https://cryptoslate.com/feed/",
}


def get_sentiment_model():
    """
    Lazily load the sentiment model only when needed.
    This prevents heavy model loading during Django startup.
    """
    global sentiment_analyzer

    if sentiment_analyzer is None:
        from transformers import pipeline  # type: ignore

        logger.info("Loading FinBERT sentiment model...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
        )

    return sentiment_analyzer


def clean_text(text):
    """
    Remove HTML tags and extra spaces from the RSS summary.
    """
    clean = re.compile(r"<.*?>")
    text_without_html = re.sub(clean, "", text or "")
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
            logger.warning("Failed to fetch from %s: %s", source, e)

    return all_articles


def get_coin_keywords(coin_symbol):
    """
    Return common keywords for a coin.
    """
    base = coin_symbol.lower()

    name_map = {
        "btc": ["btc", "bitcoin"],
        "eth": ["eth", "ethereum"],
        "ada": ["ada", "cardano"],
        "doge": ["doge", "dogecoin"],
        "dot": ["dot", "polkadot"],
        "ltc": ["ltc", "litecoin"],
        "bnb": ["bnb", "binance coin", "binance"],
        "sol": ["sol", "solana"],
        "xrp": ["xrp", "ripple"],
    }

    return name_map.get(base, [base])


def score_article_sentiment(article):
    """
    Score one article using the lazily loaded sentiment model.
    """
    model = get_sentiment_model()

    text_for_ai = f"{article.get('title', '')}. {article.get('summary', '')}".strip()
    text_for_ai = text_for_ai[:500]

    result = model(text_for_ai)[0]

    raw_label = str(result.get("label", "neutral")).lower()
    confidence = float(result.get("score", 0.0))

    if raw_label == "positive":
        sentiment_label = "BULLISH"
        sentiment_score = confidence
    elif raw_label == "negative":
        sentiment_label = "BEARISH"
        sentiment_score = -confidence
    else:
        sentiment_label = "NEUTRAL"
        sentiment_score = 0.0

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": round(float(sentiment_score), 3),
        "model_confidence": round(confidence, 3),
    }


def build_summary(label, article_count, articles):
    """
    Build a readable summary for the dashboard.
    """
    if article_count == 0:
        return "No meaningful recent news was detected."

    if label == "BULLISH":
        opening = "Recent news flow is supportive overall."
    elif label == "BEARISH":
        opening = "Recent news flow is negative overall."
    else:
        opening = "Recent news flow is mixed and not strongly directional."

    top_titles = [article.get("title", "") for article in articles[:2] if article.get("title", "")]
    if top_titles:
        return (
            f"{opening} {article_count} relevant article(s) were detected. "
            f"Key headlines include: {' | '.join(top_titles)}"
        )

    return f"{opening} {article_count} relevant article(s) were detected."


def analyze_coin_sentiment(coin_symbol):
    """
    Filter news for a specific coin and calculate the overall AI sentiment score.
    Returns a dashboard-ready structure.
    """
    articles = fetch_latest_news()
    keywords = get_coin_keywords(coin_symbol)

    relevant_articles = []
    total_score = 0.0

    for article in articles:
        text_to_search = f"{article.get('title', '')} {article.get('summary', '')}".lower()

        if any(keyword in text_to_search for keyword in keywords):
            sentiment_info = score_article_sentiment(article)

            enriched_article = {
                **article,
                **sentiment_info,
            }

            total_score += sentiment_info["sentiment_score"]
            relevant_articles.append(enriched_article)

    article_count = len(relevant_articles)

    if article_count == 0:
        return {
            "score": 0.0,
            "label": "NEUTRAL",
            "article_count": 0,
            "summary": "No meaningful recent news was detected.",
            "articles": [],
        }

    avg_score = total_score / article_count

    if avg_score > 0.20:
        final_label = "BULLISH"
    elif avg_score < -0.20:
        final_label = "BEARISH"
    else:
        final_label = "NEUTRAL"

    summary = build_summary(final_label, article_count, relevant_articles)

    return {
        "score": round(float(avg_score), 3),
        "label": final_label,
        "article_count": article_count,
        "summary": summary,
        "articles": relevant_articles,
    }


if __name__ == "__main__":
    print("Fetching and analyzing news for BTC... Please wait.")
    result = analyze_coin_sentiment("BTC")
    print(result)