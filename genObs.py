# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportGeneralTypeIssues=false
import warnings, os, sys
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, "w")

import numpy as np
import pandas as pd
import pandas_ta as ta

from data import load_data_obs


def generate_observations(symbol="AAPL", period="2y", interval="1h", start=None, end=None):
    """Fetch OHLCV and build a feature/observation matrix for an RL trading agent.

    Returns
    -------
    obs : pd.DataFrame
        Z-scored (where appropriate) feature matrix, NaNs dropped.
    df : pd.DataFrame
        Full enriched dataframe (raw indicators + 'condition' regime label).
    """
    # ---------- Fetch ----------
    df = load_data_obs(symbol, period=period, interval=interval, start=start, end=end)
    df.columns = df.columns.droplevel(1)
    df.columns = df.columns.str.lower()

    # ---------- Indicators (pandas_ta) ----------
    df.ta.adx(length=14, append=True)              # ADX_14, DMP_14, DMN_14
    df.ta.atr(length=14, append=True)              # ATRr_14
    df.ta.bbands(length=20, append=True)           # BBL/BBM/BBU/BBB/BBP
    df.ta.rsi(length=14, append=True)              # RSI_14
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.obv(append=True)

    def col(prefix):
        return next(c for c in df.columns if c.startswith(prefix))

    adx_c   = col('ADX_')
    dmp_c   = col('DMP_')
    dmn_c   = col('DMN_')
    atr_c   = col('ATRr_')
    bbb_c   = col('BBB_')
    bbp_c   = col('BBP_')
    rsi_c   = col('RSI_')
    macdh_c = col('MACDh_')

    # ---------- Derived features ----------
    df['ret_1']  = df['close'].pct_change(1)
    df['ret_5']  = df['close'].pct_change(5)
    df['ret_20'] = df['close'].pct_change(20)

    log_ret = np.log(df['close']).diff()
    df['rv_20'] = log_ret.rolling(20).std()

    df['atr_pct'] = df[atr_c] / df['close']

    df['ema50_slope']  = df['EMA_50'].pct_change(5)
    df['ema200_slope'] = df['EMA_200'].pct_change(20)
    df['ema_spread']   = (df['EMA_50'] - df['EMA_200']) / df['close']

    df['obv_z'] = (df['OBV'] - df['OBV'].rolling(100).mean()) / df['OBV'].rolling(100).std()

    df['bbw_pct_rank'] = df[bbb_c].rolling(100).rank(pct=True)

    df['ret_acf1'] = log_ret.rolling(50).apply(lambda x: x.autocorr(lag=1), raw=False)

    df['hour'] = df.index.hour

    # ---------- Regime classification ----------
    def market_condition(row):
        adx = row[adx_c]
        bbw_rank = row['bbw_pct_rank']
        if pd.isna(adx) or pd.isna(bbw_rank):
            return 'unknown'
        if adx > 25:
            return 'trending'
        if adx < 20 and bbw_rank < 0.3:
            return 'ranging'
        return 'transitional'

    df['condition'] = df.apply(market_condition, axis=1)

    # ---------- Observation vector ----------
    obs_cols = [
        'ret_1', 'ret_5', 'ret_20',
        'rv_20', 'atr_pct',
        adx_c, dmp_c, dmn_c,
        rsi_c, macdh_c,
        'ema50_slope', 'ema200_slope', 'ema_spread',
        bbp_c, 'bbw_pct_rank',
        'obv_z', 'ret_acf1',
        'hour',
    ]

    obs = df[obs_cols].copy()
    bounded = {'bbw_pct_rank', bbp_c, 'hour', dmp_c, dmn_c, rsi_c}
    for c in obs.columns:
        if c in bounded:
            continue
        mu = obs[c].rolling(500, min_periods=50).mean()
        sd = obs[c].rolling(500, min_periods=50).std()
        obs[c] = (obs[c] - mu) / sd

    obs = obs.dropna()
    return obs, df


if __name__ == "__main__":
    obs, df = generate_observations("AAPL", "2y", "1h")

    print("Regime distribution:")
    print(df['condition'].value_counts(), "\n")

    print("Last 5 observation rows:")
    print(obs.tail(5).round(3))

    print(f"\nObservation shape: {obs.shape}  (features: {obs.shape[1]})")
