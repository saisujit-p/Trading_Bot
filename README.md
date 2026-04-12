# PPO Stock Trading Bot

A reinforcement learning trading system that trains a PPO (Proximal Policy Optimization) agent on historical stock data and deploys it for live paper trading through the Alpaca API, with a real-time Plotly Dash dashboard.

## Architecture

```
data.py          Fetch OHLCV data from Yahoo Finance
    |
genObs.py        Compute technical indicators & build z-scored feature matrix
    |
tradingsys.py    Core trading simulation (portfolio tracking, rewards, costs)
    |
env.py           Gymnasium wrapper (MarketContinuous) with continuous action space
    |
training.py      Train PPO agent via Stable-Baselines3, evaluate in/out-of-sample
    |
deployment_env.py    Live market environment (paper + Alpaca broker integration)
    |
deployment.py        Live trading loop + real-time Dash dashboard
```

## Features

### Observation Space (18 features)
- **Returns**: 1-bar, 5-bar, 20-bar price returns
- **Volatility**: 20-bar realized volatility, ATR as % of price
- **Trend**: ADX, DMP, DMN, EMA-50/200 slopes and spread
- **Momentum**: RSI, MACD histogram
- **Bollinger**: Band position, bandwidth percentile rank
- **Volume**: OBV z-score
- **Autocorrelation**: 50-bar return autocorrelation (lag 1)
- **Time**: Hour of day

Non-bounded features are z-scored with a 500-bar rolling window to keep inputs stationary.

### Action Space
Continuous 2D action `[-1, 1] x [-1, 1]`:
- **Dimension 1**: Direction — sell (< -0.6), hold (-0.6 to 0.6), buy (> 0.6)
- **Dimension 2**: Fraction — mapped to [0, 1], controlling what % of cash to spend or shares to sell

### Reward Design
- **Primary signal**: Log return of portfolio value (bar-over-bar)
- **Turnover penalty**: -0.001 * turnover ratio (discourages excessive trading)
- **Invalid action penalty**: -0.01 for attempting to buy with no cash or sell with no shares
- **Terminal bonus**: 10% of total return added at episode end

### Live Deployment
- Trades through Alpaca paper trading API
- Executes one trade per bar interval (default: 1 hour)
- Syncs cash/shares from broker after every order
- Automatically waits for market open, resumes across trading days
- Real-time dashboard at `http://127.0.0.1:8050` showing:
  - Stock price with BUY/SELL markers
  - Portfolio value over time
  - Current equity displayed while market is closed

## Setup

### Prerequisites
```
pip install stable-baselines3 gymnasium yfinance pandas_ta alpaca-py dash plotly
```

### Alpaca Credentials
Create a `creds.txt` file with your [Alpaca paper trading](https://app.alpaca.markets/paper/dashboard/overview) keys:
```
API_KEY: your_api_key_here
Secret: your_secret_key_here
```

Or set environment variables:
```bash
export ALPACA_API_KEY=your_api_key_here
export ALPACA_SECRET_KEY=your_secret_key_here
```

## Usage

### Train a model
```bash
cd V1
python training.py
```
Trains a PPO agent on 80% of historical data (default: AAPL, 10y, 1d bars) and evaluates on the held-out 20%. Saves the model as `TimCook`.

### Deploy live (paper trading)
```bash
cd V1
python deployment.py
```
Opens a dashboard at [http://127.0.0.1:8050](http://127.0.0.1:8050). The bot waits for the market to open, then trades AAPL on 1-hour bars until close. Resumes automatically the next trading day.

### Configuration
Edit the constants at the top of `deployment.py`:
```python
SYMBOL       = "AAPL"      # Ticker to trade
INTERVAL     = "1h"        # Bar interval: 1m, 5m, 15m, 30m, 1h, 1d
MODEL_PATH   = "TimCook"  # Path to trained model
POLL_SECONDS = 60           # How often to check market status
```

## Project Structure

| File | Description |
|------|-------------|
| `data.py` | Yahoo Finance data fetching utilities |
| `genObs.py` | Technical indicator computation and feature engineering |
| `tradingsys.py` | Core trading environment with transaction costs and slippage |
| `env.py` | Gymnasium-compatible environment with train/test splits |
| `training.py` | PPO training and in-sample/out-of-sample evaluation |
| `deployment_env.py` | Live market wrapper + Alpaca broker integration |
| `deployment.py` | Live trading loop with Dash dashboard |
| `Testing_new_Market.py` | Manual testing script for the model |
