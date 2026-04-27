import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .models import MarketCandle

BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

COINS = ["BTC", "ETH", "BNB", "ADA", "DOGE", "DOT", "LTC"]
QUOTE_ASSET = "USDT"

DEFAULT_INTERVAL = "1h"
DEFAULT_TRAINING_MONTHS = 6
DEFAULT_LIMIT = 1000

INTERVAL_TO_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

RANGE_TO_DELTA = {
    "1d": timedelta(days=1),
    "1w": timedelta(days=7),
    "1m": timedelta(days=30),
    "3m": timedelta(days=90),
    "6m": timedelta(days=180),
}


def get_binance_symbol(symbol: str) -> str:
    return f"{symbol.upper()}{QUOTE_ASSET}"


def interval_to_timedelta(interval: str) -> timedelta:
    if interval == "1h":
        return timedelta(hours=1)
    if interval == "4h":
        return timedelta(hours=4)
    if interval == "1d":
        return timedelta(days=1)
    raise ValueError(f"Unsupported interval: {interval}")


def range_to_start_datetime(range_key: str) -> datetime:
    now = datetime.now(timezone.utc)
    delta = RANGE_TO_DELTA.get(range_key, timedelta(days=180))
    return now - delta


def months_ago_datetime(months: int) -> datetime:
    now = datetime.now(timezone.utc)
    return now - timedelta(days=30 * months)


def klines_to_dataframe(symbol: str, interval: str, klines: list) -> pd.DataFrame:
    rows = []

    for item in klines:
        rows.append(
            {
                "time": datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volumefrom": float(item[5]),
                "volumeto": float(item[7]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "conversionType": "direct",
                "conversionSymbol": "",
                "interval": interval,
                "symbol": symbol.upper(),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def fetch_coin_data(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = 200,
) -> pd.DataFrame:
    pair = get_binance_symbol(symbol)

    params = {
        "symbol": pair,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        f"{BASE_URL}{KLINES_ENDPOINT}",
        params=params,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError(f"No Binance data found for {symbol} {interval}")

    return klines_to_dataframe(symbol=symbol, interval=interval, klines=data)


def fetch_coin_data_range(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit_per_call: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    pair = get_binance_symbol(symbol)

    if start_time is None:
        start_time = months_ago_datetime(DEFAULT_TRAINING_MONTHS)

    if end_time is None:
        end_time = datetime.now(timezone.utc)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": pair,
            "interval": interval,
            "limit": limit_per_call,
            "startTime": current_start,
            "endTime": end_ms,
        }

        response = requests.get(
            f"{BASE_URL}{KLINES_ENDPOINT}",
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        batch = response.json()

        if not isinstance(batch, list) or not batch:
            break

        all_klines.extend(batch)

        last_open_time = int(batch[-1][0])
        next_open_time = last_open_time + INTERVAL_TO_MS[interval]

        if next_open_time <= current_start:
            break

        current_start = next_open_time

        if len(batch) < limit_per_call:
            break

    if not all_klines:
        return pd.DataFrame()

    df = klines_to_dataframe(symbol=symbol, interval=interval, klines=all_klines)
    return df


def fetch_all_coins(
    interval: str = DEFAULT_INTERVAL,
    months: int = DEFAULT_TRAINING_MONTHS,
) -> pd.DataFrame:
    all_data = []
    start_time = months_ago_datetime(months)

    for coin in COINS:
        try:
            df = fetch_coin_data_range(
                symbol=coin,
                interval=interval,
                start_time=start_time,
            )
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error fetching {coin} {interval}: {e}")

    if not all_data:
        raise Exception("No Binance API data available for any coin")

    return pd.concat(all_data, ignore_index=True).sort_values(["symbol", "time"]).reset_index(drop=True)


def save_market_candles(df: pd.DataFrame, interval: str | None = None) -> int:
    if df is None or df.empty:
        return 0

    created_or_updated = 0

    for _, row in df.iterrows():
        candle_interval = interval or row.get("interval", DEFAULT_INTERVAL)

        MarketCandle.objects.update_or_create(
            symbol=row["symbol"],
            interval=candle_interval,
            time=row["time"],
            defaults={
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", row.get("volumefrom", 0.0))),
                "quote_volume": float(row.get("quote_volume", row.get("volumeto", 0.0))),
            },
        )
        created_or_updated += 1

    return created_or_updated


def build_history_payload_from_df(df: pd.DataFrame) -> list[dict]:
    rows = []

    if df is None or df.empty:
        return rows

    for _, row in df.iterrows():
        rows.append(
            {
                "time": int(pd.Timestamp(row["time"]).timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", row.get("volumefrom", 0.0))),
                "quote_volume": float(row.get("quote_volume", row.get("volumeto", 0.0))),
            }
        )

    return rows


def get_market_history(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    range_key: str = "6m",
    prefer_db: bool = True,
) -> list[dict]:
    symbol = symbol.upper()
    start_dt = range_to_start_datetime(range_key)

    if prefer_db:
        qs = (
            MarketCandle.objects.filter(
                symbol=symbol,
                interval=interval,
                time__gte=start_dt,
            )
            .order_by("time")
            .values(
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
            )
        )

        rows = list(qs)
        if rows:
            payload = []
            for row in rows:
                payload.append(
                    {
                        "time": int(row["time"].timestamp()),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "quote_volume": float(row["quote_volume"]),
                    }
                )
            return payload

    df = fetch_coin_data_range(
        symbol=symbol,
        interval=interval,
        start_time=start_dt,
    )

    if not df.empty:
        save_market_candles(df, interval=interval)

    return build_history_payload_from_df(df)


def seed_history_for_all_coins(
    intervals: list[str] | None = None,
    months: int = DEFAULT_TRAINING_MONTHS,
) -> dict:
    if intervals is None:
        intervals = ["1h", "4h", "1d"]

    summary = {}

    for interval in intervals:
        try:
            df = fetch_all_coins(interval=interval, months=months)
            count = save_market_candles(df, interval=interval)
            summary[interval] = {
                "rows": int(len(df)),
                "saved": int(count),
            }
        except Exception as e:
            summary[interval] = {
                "rows": 0,
                "saved": 0,
                "error": str(e),
            }

    return summary