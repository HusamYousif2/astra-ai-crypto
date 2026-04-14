# # market/services.py

# import requests
# import pandas as pd
# from datetime import datetime

# CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

# COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "LTC", "TRX"]


# def fetch_coin_data(symbol, limit=200):
#     """
#     Fetch historical hourly data for a single coin.
#     """
#     params = {
#         "fsym": symbol,
#         "tsym": "USD",
#         "limit": limit
#     }

#     response = requests.get(CRYPTOCOMPARE_URL, params=params)

#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch data for {symbol}")

#     data = response.json()["Data"]["Data"]

#     df = pd.DataFrame(data)

#     # Convert timestamp
#     df["time"] = df["time"].apply(lambda x: datetime.fromtimestamp(x))

#     df["symbol"] = symbol

#     return df


# def fetch_all_coins():
#     """
#     Fetch data for all coins and merge into one DataFrame.
#     """
#     all_data = []

#     for coin in COINS:
#         try:
#             df = fetch_coin_data(coin)
#             all_data.append(df)
#         except Exception as e:
#             print(f"Error fetching {coin}: {e}")

#     final_df = pd.concat(all_data, ignore_index=True)

#     return final_df

# market/services.py

import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

COINS = ["BTC", "ETH", "BNB", "ADA", "DOGE", "DOT", "LTC"]
QUOTE_ASSET = "USDT"
INTERVAL = "1h"
LIMIT = 200


def fetch_coin_data(symbol):
    """
    Fetch hourly OHLCV market data for a single coin from Binance Spot API.
    """
    pair = f"{symbol}{QUOTE_ASSET}"

    params = {
        "symbol": pair,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    response = requests.get(
        f"{BASE_URL}{KLINES_ENDPOINT}",
        params=params,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or not data:
        print(f"No historical data found for {symbol}")
        return None

    rows = []
    for item in data:
        rows.append(
            {
                "time": datetime.fromtimestamp(item[0] / 1000),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volumefrom": float(item[5]),
                "volumeto": float(item[7]),
                "conversionType": "direct",
                "conversionSymbol": "",
                "symbol": symbol,
            }
        )

    df = pd.DataFrame(rows)
    return df


def fetch_all_coins():
    """
    Fetch market data for all configured coins and merge into one DataFrame.
    """
    all_data = []

    for coin in COINS:
        try:
            df = fetch_coin_data(coin)
            if df is not None:
                all_data.append(df)
        except Exception as e:
            print(f"Error fetching {coin}: {e}")

    if not all_data:
        raise Exception("No Binance API data available for any coin")

    return pd.concat(all_data, ignore_index=True)