"""
Market regime detection using SMA crossover + price position.

Regimes:
  - bull:     SMA50 > SMA200 and price > SMA200 (confirmed uptrend)
  - bear:     SMA50 < SMA200 and price < SMA200 (confirmed downtrend)
  - sideways: everything else (transitional / range-bound)

All signals are causal — label at time t only uses prices[:t+1].
"""
import numpy as np


def _sma(prices, window):
    """Simple moving average. Returns NaN for positions without full window."""
    prices = prices.astype(np.float64)
    out = np.full(len(prices), np.nan, dtype=np.float64)
    cs = np.cumsum(prices)
    out[window - 1:] = (cs[window - 1:] - np.concatenate([[0.0], cs[:-window]])) / window
    return out


def detect_regime(prices, sma_short=50, sma_long=200):
    """
    Label each timestep as 'bull', 'bear', or 'sideways'.

    For the first sma_long-1 steps where SMA200 isn't available,
    falls back to price vs SMA50 (or 'sideways' if neither available).
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    sma50 = _sma(prices, sma_short)
    sma200 = _sma(prices, sma_long)

    labels = np.array(['sideways'] * n, dtype='U8')

    for i in range(n):
        if np.isnan(sma200[i]):
            if not np.isnan(sma50[i]):
                if prices[i] > sma50[i]:
                    labels[i] = 'bull'
                elif prices[i] < sma50[i]:
                    labels[i] = 'bear'
        else:
            if sma50[i] > sma200[i] and prices[i] > sma200[i]:
                labels[i] = 'bull'
            elif sma50[i] < sma200[i] and prices[i] < sma200[i]:
                labels[i] = 'bear'

    return labels


def split_by_regime(prices, features, labels, target_regime, min_length=30):
    """
    Extract contiguous segments where labels == target_regime.
    Returns list of (prices_segment, features_segment) tuples.
    Segments shorter than min_length trading days are discarded.
    """
    segments = []
    start = None
    n = len(labels)

    for i in range(n):
        if labels[i] == target_regime:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_length:
                    segments.append((prices[start:i].copy(), features[start:i].copy()))
                start = None

    if start is not None and n - start >= min_length:
        segments.append((prices[start:].copy(), features[start:].copy()))

    return segments


def get_current_regime(prices, sma_short=50, sma_long=200):
    """Detect the regime at the most recent timestep."""
    labels = detect_regime(prices, sma_short, sma_long)
    return str(labels[-1])


def regime_summary(prices):
    """Print distribution of regimes across the price series."""
    labels = detect_regime(prices)
    n = len(labels)
    for r in ('bull', 'bear', 'sideways'):
        count = np.sum(labels == r)
        print(f"  {r:>8s}: {count:5d} days ({100*count/n:.1f}%)")
    return labels
