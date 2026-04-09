import yfinance as yf
import numpy as np

'''
def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    prices = data["Close"].to_numpy(dtype=np.float32).reshape(-1)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 2:
        raise ValueError(f"Not enough valid price data loaded for {symbol}.")
    return prices
'''
def load_data(symbol, period,interval):
    data = yf.download(symbol, period=period, interval=interval)
    prices = data["Close"].to_numpy(dtype=np.float32).reshape(-1)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size < 2:
        raise ValueError(f"Not enough valid price data loaded for {symbol}.")
    return prices

def load_data_obs(symbol, period=None, interval="1d", start=None, end=None):
    if start is not None or end is not None:
        data = yf.download(symbol, start=start, end=end, interval=interval)
    else:
        data = yf.download(symbol, period=period, interval=interval)
    return data