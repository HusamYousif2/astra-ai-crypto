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

URL = "https://min-api.cryptocompare.com/data/v2/histohour"

COINS = ["BTC","ETH","BNB","ADA","DOGE","DOT","LTC"]

# 1.api key
API_KEY = "f9dc9aa774f441add9554c80a1b6cce0d457661c75e8594ef8097a8459fe6921"

def fetch_coin_data(symbol):
    # 2.api key
    headers = {
        "authorization": f"Apikey {API_KEY}"
    }
    
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": 200
    }

    # 3. api key with the header
    response = requests.get(URL, headers=headers, params=params)

    data = response.json()

    
    if "Data" not in data or "Data" not in data.get("Data", {}):
        error_message = data.get("Message", "Unknown API Error")
        print(f"Error fetching {symbol}. Reason from API: {error_message}")
        return None

    data = data["Data"]["Data"]

    if not data:
        print(f"No historical data found for {symbol}")
        return None

    df = pd.DataFrame(data)

    df["time"] = df["time"].apply(lambda x: datetime.fromtimestamp(x))
    df["symbol"] = symbol

    return df


def fetch_all_coins():
    all_data = []

    for coin in COINS:
        df = fetch_coin_data(coin)

        if df is not None:
            all_data.append(df)

    if not all_data:
        raise Exception("No API data available for any coin")

    return pd.concat(all_data, ignore_index=True)