"""
Live deployment loop + Plotly Dash dashboard.

Trading thread:
    Waits for the Alpaca market to open, then each bar predicts an action with
    the trained PPO model and routes it through AlpacaLiveMarket. Stops when
    the market closes.

Dashboard (http://127.0.0.1:8050):
    - Top chart:    stock price with BUY / SELL markers
    - Bottom chart: portfolio value over time
    Refreshes every few seconds.
"""

import os
import time
import threading


def _load_creds(path=os.path.join(os.path.dirname(__file__), "creds.txt")):
    """Load API_KEY / Secret from creds.txt into os.environ if not already set."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip().lower(), val.strip()
            if key == "api_key":
                os.environ.setdefault("ALPACA_API_KEY", val)
            elif key == "secret":
                os.environ.setdefault("ALPACA_SECRET_KEY", val)


_load_creds()

from stable_baselines3 import PPO
from alpaca.trading.client import TradingClient

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deployment_env import AlpacaLiveMarket

# ----------------------------- config ---------------------------------
SYMBOL       = "AAPL"
INTERVAL     = "1h"
MODEL_PATH   = "Nigesh_AAPL_V2"
POLL_SECONDS = 60
BAR_SECONDS  = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                "1h": 3600, "1d": 86400}[INTERVAL]

# ------------------------ shared state (thread-safe-ish) --------------
STATE = {
    "times":     [],
    "prices":    [],
    "portfolio": [],
    "actions":   [],   # list of (time, price, "BUY"/"SELL")
    "status":    "starting",
    "frozen_equity": None,  # last known account equity while market is closed
}
LOCK = threading.Lock()


_TRADING_CLIENT = None


def _trading_client():
    global _TRADING_CLIENT
    if _TRADING_CLIENT is None:
        _TRADING_CLIENT = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
    return _TRADING_CLIENT


def get_clock():
    return _trading_client().get_clock()


def fetch_equity():
    """Return current Alpaca account equity, or None on failure."""
    try:
        acct = _trading_client().get_account()
        return float(getattr(acct, "equity", 0) or 0)
    except Exception as e:
        print(f"[fetch_equity] {e}")
        return None


def wait_for_open():
    while True:
        try:
            clock = get_clock()
        except Exception as e:
            with LOCK:
                STATE["status"] = f"clock error: {e}"
            print(f"[wait_for_open] clock error: {e}")
            time.sleep(POLL_SECONDS)
            continue
        if getattr(clock, "is_open", False):
            with LOCK:
                STATE["status"] = "market open"
            return
        next_open = getattr(clock, "next_open", None)
        equity = fetch_equity()
        with LOCK:
            if equity is not None:
                STATE["frozen_equity"] = equity
            STATE["status"] = f"closed — next open {next_open}"
        time.sleep(POLL_SECONDS)


def _next_bar_target():
    """Return the unix-time of the next bar close + small buffer."""
    now = time.time()
    return (int(now // BAR_SECONDS) + 1) * BAR_SECONDS + 5


def trading_session(model):
    """One open->close trading session. Returns when the market closes."""
    live = AlpacaLiveMarket(symbol=SYMBOL, interval=INTERVAL)
    obs = live.reset()

    with LOCK:
        STATE["status"] = "trading"

    while True:
        try:
            clock = get_clock()
        except Exception as e:
            with LOCK:
                STATE["status"] = f"clock error: {e}"
            print(f"[trading_session] clock error: {e}")
            time.sleep(POLL_SECONDS)
            continue

        if not getattr(clock, "is_open", False):
            with LOCK:
                STATE["status"] = "market closed"
            return

        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, info = live.step(action)
        except Exception as e:
            with LOCK:
                STATE["status"] = f"step error: {e}"
            print(f"[trading_session] step error: {e}")
            time.sleep(POLL_SECONDS)
            continue

        with LOCK:
            STATE["times"].append(info["time"])
            STATE["prices"].append(info["price"])
            STATE["portfolio"].append(info["portfolio_value"])
            if info["action"] in ("BUY", "SELL"):
                STATE["actions"].append((info["time"], info["price"], info["action"]))

        # Sleep until the next bar boundary, waking early if market closes.
        target = _next_bar_target()
        while time.time() < target:
            remaining = target - time.time()
            time.sleep(max(0.1, min(POLL_SECONDS, remaining)))
            try:
                if not getattr(get_clock(), "is_open", False):
                    break
            except Exception as e:
                print(f"[trading_session] clock poll error: {e}")


def trading_loop():
    """Run trading sessions forever, sleeping through closed periods."""
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        with LOCK:
            STATE["status"] = f"model load error: {e}"
        print(f"[trading_loop] model load error: {e}")
        return

    while True:
        with LOCK:
            STATE["status"] = "waiting for open"
        wait_for_open()
        try:
            trading_session(model)
        except Exception as e:
            with LOCK:
                STATE["status"] = f"session crashed: {e}"
            print(f"[trading_loop] session crashed: {e}")
            time.sleep(POLL_SECONDS)


# ------------------------------ Dash app ------------------------------
app = dash.Dash(__name__)
app.title = f"{SYMBOL} Live PPO"

app.layout = html.Div(
    style={"fontFamily": "monospace", "maxWidth": "1100px", "margin": "auto"},
    children=[
        html.H2(f"{SYMBOL} — Live PPO Paper Trading"),
        html.Div(id="status", style={"marginBottom": "10px", "color": "#555"}),
        dcc.Graph(id="chart"),
        dcc.Interval(id="tick", interval=5000, n_intervals=0),
    ],
)


@app.callback(
    [Output("chart", "figure"), Output("status", "children")],
    [Input("tick", "n_intervals")],
)
def update(_):
    with LOCK:
        times     = list(STATE["times"])
        prices    = list(STATE["prices"])
        portfolio = list(STATE["portfolio"])
        actions   = list(STATE["actions"])
        status    = STATE["status"]
        frozen    = STATE["frozen_equity"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"{SYMBOL} price", "Portfolio value ($)"),
    )

    fig.add_trace(
        go.Scatter(x=times, y=prices, mode="lines", name="price",
                   line=dict(color="#1f77b4")),
        row=1, col=1,
    )

    buys  = [(t, p) for (t, p, a) in actions if a == "BUY"]
    sells = [(t, p) for (t, p, a) in actions if a == "SELL"]
    if buys:
        bt, bp = zip(*buys)
        fig.add_trace(
            go.Scatter(x=bt, y=bp, mode="markers", name="BUY",
                       marker=dict(symbol="triangle-up", color="green", size=12)),
            row=1, col=1,
        )
    if sells:
        st, sp = zip(*sells)
        fig.add_trace(
            go.Scatter(x=st, y=sp, mode="markers", name="SELL",
                       marker=dict(symbol="triangle-down", color="red", size=12)),
            row=1, col=1,
        )

    if portfolio:
        fig.add_trace(
            go.Scatter(x=times, y=portfolio, mode="lines", name="portfolio",
                       line=dict(color="#2ca02c")),
            row=2, col=1,
        )
    elif frozen is not None:
        # Market closed and no live bars yet — show the current account
        # equity as a flat frozen line so the user sees their portfolio.
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[frozen, frozen], mode="lines",
                       name=f"portfolio (frozen ${frozen:,.2f})",
                       line=dict(color="#2ca02c", dash="dash")),
            row=2, col=1,
        )
        fig.update_yaxes(range=[frozen * 0.99, frozen * 1.01], row=2, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)

    fig.update_layout(height=700, margin=dict(l=60, r=20, t=50, b=40),
                      showlegend=True, template="plotly_white")
    equity_str = f"  |  equity: ${frozen:,.2f}" if frozen is not None else ""
    return fig, f"Status: {status}  |  bars recorded: {len(times)}{equity_str}"


# --------------------------------- main -------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    app.run(debug=False, host="127.0.0.1", port=8050)
