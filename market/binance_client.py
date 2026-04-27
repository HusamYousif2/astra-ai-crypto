import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


BINANCE_BASE_URL = "https://api.binance.com"


SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "DOT": "DOTUSDT",
    "LTC": "LTCUSDT",
}


INTERVAL_TO_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def to_binance_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()
    return SYMBOL_MAP.get(symbol, f"{symbol}USDT")


def interval_to_timedelta(interval: str) -> timedelta:
    if interval == "1h":
        return timedelta(hours=1)
    if interval == "4h":
        return timedelta(hours=4)
    if interval == "1d":
        return timedelta(days=1)
    raise ValueError(f"Unsupported interval: {interval}")


def datetime_to_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_klines(symbol: str, interval: str = "1h", start_time: datetime = None, end_time: datetime = None, limit: int = 1000):
    params = {
        "symbol": to_binance_symbol(symbol),
        "interval": interval,
        "limit": limit,
    }

    if start_time is not None:
        params["startTime"] = datetime_to_millis(start_time)

    if end_time is not None:
        params["endTime"] = datetime_to_millis(end_time)

    response = requests.get(f"{BINANCE_BASE_URL}/api/v3/klines", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def klines_to_dataframe(klines, symbol: str, interval: str) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(columns=["symbol", "interval", "timestamp", "open", "high", "low", "close", "volume"])

    rows = []
    for item in klines:
        rows.append({
            "symbol": symbol.upper(),
            "interval": interval,
            "timestamp": datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset=["symbol", "interval", "timestamp"]).reset_index(drop=True)
    return df


def fetch_historical_klines(symbol: str, interval: str = "1h", months: int = 6) -> pd.DataFrame:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30 * months)

    all_klines = []
    current_start = start_time
    step_ms = INTERVAL_TO_MS[interval]

    while current_start < end_time:
        klines = get_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time,
            limit=1000,
        )

        if not klines:
            break

        all_klines.extend(klines)

        last_open_time_ms = klines[-1][0]
        next_start_ms = last_open_time_ms + step_ms
        current_start = datetime.fromtimestamp(next_start_ms / 1000, tz=timezone.utc)

        time.sleep(0.15)

        if len(klines) < 1000:
            break

    return klines_to_dataframe(all_klines, symbol=symbol, interval=interval)


def fetch_recent_klines(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    klines = get_klines(symbol=symbol, interval=interval, limit=limit)
    return klines_to_dataframe(klines, symbol=symbol, interval=interval)


def fetch_latest_price(symbol: str) -> dict:
    response = requests.get(
        f"{BINANCE_BASE_URL}/api/v3/ticker/24hr",
        params={"symbol": to_binance_symbol(symbol)},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "symbol": symbol.upper(),
        "last_price": float(data["lastPrice"]),
        "price_change_percent": float(data["priceChangePercent"]),
        "high_price": float(data["highPrice"]),
        "low_price": float(data["lowPrice"]),
        "volume": float(data["volume"]),
        "quote_volume": float(data["quoteVolume"]),
    }