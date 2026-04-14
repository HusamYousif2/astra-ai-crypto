import re
import numpy as np
import pandas as pd

from ..nlp_engine import analyze_coin_sentiment


NEWS_IMPACT_LABELS = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
}

NEWS_RELEVANCE_LABELS = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
}

EVENT_URGENCY_KEYWORDS = [
    "breakout",
    "surge",
    "crash",
    "etf",
    "ban",
    "hack",
    "lawsuit",
    "liquidation",
    "whale",
    "approval",
    "rejection",
    "halt",
    "outflow",
    "inflow",
    "ceasefire",
    "war",
    "conflict",
    "rates",
    "fed",
    "sec",
]

COIN_KEYWORD_MAP = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "BNB": ["bnb", "binance coin", "binance"],
    "ADA": ["ada", "cardano"],
    "DOGE": ["doge", "dogecoin"],
    "DOT": ["dot", "polkadot"],
    "LTC": ["ltc", "litecoin"],
    "XRP": ["xrp", "ripple"],
    "SOL": ["sol", "solana"],
}


def normalize_text(text):
    """
    Normalize text for keyword matching.
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_coin_keywords(coin_symbol):
    """
    Return standard keywords for an asset.
    """
    return COIN_KEYWORD_MAP.get(coin_symbol.upper(), [coin_symbol.lower()])


def compute_article_relevance(article, coin_symbol):
    """
    Estimate how relevant an article is to the selected asset.
    """
    keywords = get_coin_keywords(coin_symbol)

    title = normalize_text(article.get("title", ""))
    summary = normalize_text(article.get("summary", ""))

    relevance_score = 0.0

    for keyword in keywords:
        if keyword in title:
            relevance_score += 2.0
        if keyword in summary:
            relevance_score += 1.0

    if relevance_score >= 2.5:
        return 1.0
    if relevance_score >= 1.0:
        return 0.65
    return 0.25


def compute_article_urgency(article):
    """
    Estimate whether the article describes an urgent catalyst.
    """
    title = normalize_text(article.get("title", ""))
    summary = normalize_text(article.get("summary", ""))
    combined = f"{title} {summary}"

    score = 0.0

    for keyword in EVENT_URGENCY_KEYWORDS:
        if keyword in combined:
            score += 1.0

    if score >= 2:
        return 1.0
    if score >= 1:
        return 0.6
    return 0.2


def compute_article_impact(article, coin_symbol):
    """
    Estimate article impact by combining sentiment, relevance, and urgency.
    """
    sentiment_score = abs(article.get("sentiment_score", 0.0))
    model_confidence = article.get("model_confidence", 0.0)

    relevance = compute_article_relevance(article, coin_symbol)
    urgency = compute_article_urgency(article)

    impact_score = (
        (sentiment_score * 0.40)
        + (model_confidence * 0.20)
        + (relevance * 0.25)
        + (urgency * 0.15)
    )

    impact_score = round(float(min(max(impact_score, 0.0), 1.0)), 3)

    if impact_score >= 0.70:
        impact_label = NEWS_IMPACT_LABELS["HIGH"]
    elif impact_score >= 0.40:
        impact_label = NEWS_IMPACT_LABELS["MEDIUM"]
    else:
        impact_label = NEWS_IMPACT_LABELS["LOW"]

    return {
        "impact_score": impact_score,
        "impact_label": impact_label,
        "relevance_score": round(float(relevance), 3),
        "urgency_score": round(float(urgency), 3),
    }


def classify_news_relevance(articles):
    """
    Classify overall relevance of the current news set.
    """
    if not articles:
        return NEWS_RELEVANCE_LABELS["LOW"]

    relevance_values = [article.get("relevance_score", 0.0) for article in articles]
    avg_relevance = float(np.mean(relevance_values)) if relevance_values else 0.0

    if avg_relevance >= 0.75:
        return NEWS_RELEVANCE_LABELS["HIGH"]
    if avg_relevance >= 0.45:
        return NEWS_RELEVANCE_LABELS["MEDIUM"]
    return NEWS_RELEVANCE_LABELS["LOW"]


def detect_news_market_contradiction(sentiment_label, market_state_label):
    """
    Detect contradiction between news tone and market regime.
    """
    if not market_state_label:
        return False

    bullish_states = {"Bullish Momentum"}
    bearish_states = {"Bearish Momentum"}

    if sentiment_label == "BULLISH" and market_state_label in bearish_states:
        return True

    if sentiment_label == "BEARISH" and market_state_label in bullish_states:
        return True

    return False


def summarize_top_drivers(enriched_articles):
    """
    Extract short top drivers from the most important articles.
    """
    if not enriched_articles:
        return []

    drivers = []
    seen = set()

    for article in enriched_articles[:3]:
        title = article.get("title", "").strip()
        if title and title not in seen:
            seen.add(title)
            drivers.append(title)

    return drivers[:3]


def build_news_summary_v2(
    coin_symbol,
    sentiment_label,
    article_count,
    impact_label,
    relevance_label,
    contradiction,
    top_drivers,
):
    """
    Build a business-friendly AI summary for the news layer.
    """
    if article_count == 0:
        return f"No meaningful recent news was detected for {coin_symbol.upper()}."

    parts = []

    if sentiment_label == "BULLISH":
        parts.append("Recent news flow is supportive overall.")
    elif sentiment_label == "BEARISH":
        parts.append("Recent news flow is negative overall.")
    else:
        parts.append("Recent news flow is mixed and not strongly directional.")

    parts.append(
        f"{article_count} relevant article(s) were detected with {impact_label.lower()} impact and {relevance_label.lower()} relevance."
    )

    if contradiction:
        parts.append("News tone currently conflicts with the market structure, so caution is warranted.")

    if top_drivers:
        parts.append("Key drivers include: " + " | ".join(top_drivers[:2]))

    return " ".join(parts)


def enrich_articles_v2(coin_symbol, raw_articles):
    """
    Add relevance and impact metadata to article-level records.
    """
    enriched = []

    for article in raw_articles:
        impact_data = compute_article_impact(article, coin_symbol)

        enriched_article = {
            "source": article.get("source", ""),
            "title": article.get("title", ""),
            "summary": article.get("summary", ""),
            "link": article.get("link", ""),
            "sentiment_label": article.get("sentiment_label", "NEUTRAL"),
            "sentiment_score": article.get("sentiment_score", 0.0),
            "model_confidence": article.get("model_confidence", 0.0),
            "impact_score": impact_data["impact_score"],
            "impact_label": impact_data["impact_label"],
            "relevance_score": impact_data["relevance_score"],
            "urgency_score": impact_data["urgency_score"],
        }

        enriched.append(enriched_article)

    enriched = sorted(
        enriched,
        key=lambda x: (x["impact_score"], abs(x["sentiment_score"])),
        reverse=True,
    )

    return enriched


def build_news_context_v2(coin_symbol, market_state_label=None):
    """
    Build V2 news context for the selected asset.
    """
    raw_result = analyze_coin_sentiment(coin_symbol)

    raw_articles = raw_result.get("articles", [])
    enriched_articles = enrich_articles_v2(coin_symbol, raw_articles)

    sentiment_score = raw_result.get("score", 0.0)
    sentiment_label = raw_result.get("label", "NEUTRAL")
    article_count = raw_result.get("article_count", 0)

    relevance_label = classify_news_relevance(enriched_articles)

    if enriched_articles:
        avg_impact = float(np.mean([item["impact_score"] for item in enriched_articles]))
    else:
        avg_impact = 0.0

    if avg_impact >= 0.70:
        impact_label = NEWS_IMPACT_LABELS["HIGH"]
    elif avg_impact >= 0.40:
        impact_label = NEWS_IMPACT_LABELS["MEDIUM"]
    else:
        impact_label = NEWS_IMPACT_LABELS["LOW"]

    contradiction = detect_news_market_contradiction(
        sentiment_label=sentiment_label,
        market_state_label=market_state_label,
    )

    top_drivers = summarize_top_drivers(enriched_articles)

    summary = build_news_summary_v2(
        coin_symbol=coin_symbol,
        sentiment_label=sentiment_label,
        article_count=article_count,
        impact_label=impact_label,
        relevance_label=relevance_label,
        contradiction=contradiction,
        top_drivers=top_drivers,
    )

    return {
        "coin": coin_symbol,
        "sentiment_score": round(float(sentiment_score), 3),
        "sentiment_label": sentiment_label,
        "article_count": article_count,
        "news_impact": impact_label,
        "news_relevance": relevance_label,
        "contradiction_with_market": contradiction,
        "summary": summary,
        "top_drivers": top_drivers,
        "top_articles": enriched_articles[:5],
    }


if __name__ == "__main__":
    result = build_news_context_v2("BTC", market_state_label="Bullish Momentum")
    print(result)