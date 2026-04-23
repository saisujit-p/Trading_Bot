# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
V2 live deployment: meta-controller over regime specialists, single symbol.

Trading thread:
    Waits for the market to open, then each bar:
      1. refresh price history
      2. HMM predicts current regime
      3. drawdown gate (bear→sideways when dd > -10%) — matches eval_meta.py
      4. dispatch to regime specialist → action → broker
    Stops when the market closes, then loops until next open.

Dashboard (http://127.0.0.1:8052):
    - Price chart with BUY/SELL markers + regime band strip
    - Stacked portfolio chart (cash vs. share value)
    - Live status line (regime, gate fires, dispatched specialist, exposure)

Retraining:
    Not automatic — V2 retrain is 30-45 min per specialist (~2-3h full
    pipeline). Kick off manually via overnight_training.py when desired.
"""
import os, sys, csv, time, threading
import numpy as np
import pandas as pd


def _load_creds(path=os.path.join(os.path.dirname(__file__), "creds.txt")):
    """Load API_KEY / Secret from creds.txt into os.environ if not set."""
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

from stable_baselines3 import SAC
from alpaca.trading.client import TradingClient

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hmm_regime import HMMRegimeDetector

from deployment_env import AlpacaLiveMarket, LiveMarket
from env import HMM_MODELS_DIR
from _shared import regime_model_dir
from eval_meta import (
    DRAWDOWN_GATE_THRESHOLD, DRAWDOWN_GATE_WINDOW, _drawdown_series,
)

# ----------------------------- config ---------------------------------
SYMBOL = "AAPL"
INTERVAL = "1d"                  # V2 specialists trained on daily bars
DASH_PORT = 8052                 # distinct from V1 PPO (8050) / V1 SAC (8051)
POLL_SECONDS = 60
USE_ALPACA = True                # False = local paper portfolio
INITIAL_CASH = 100.0             # only used when USE_ALPACA = False

CSV_PATH = os.path.join(os.path.dirname(__file__), "trade_log_v2.csv")

REGIMES = ("bull", "bear", "sideways")
REGIME_COLORS = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#95a5a6"}
CSV_FIELDS = ["time", "price", "portfolio_value", "cash", "share_value",
              "exposure", "regime_raw", "regime_dispatched", "action"]

# ------------------------ shared state --------------------------------
STATE = {
    "times": [],
    "prices": [],
    "portfolio": [],
    "cash": [],
    "share_value": [],
    "exposure": [],
    "regimes_dispatched": [],   # one per bar — for the regime strip chart
    "actions": [],              # list of (time, price, "BUY"/"SELL")
    "status": "starting",
    "current_regime_raw": "—",
    "current_regime_dispatched": "—",
    "gate_fires": 0,
    "frozen_equity": None,
    "stop_requested": False,
}
LOCK = threading.Lock()


# ---------- CSV persistence so charts survive restarts ----------------
def _load_state_from_csv():
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            STATE["times"].append(row["time"])
            STATE["prices"].append(float(row["price"]))
            STATE["portfolio"].append(float(row["portfolio_value"]))
            STATE["cash"].append(max(0, float(row["cash"])))
            STATE["share_value"].append(max(0, float(row["share_value"])))
            STATE["exposure"].append(float(row.get("exposure", 0) or 0))
            STATE["regimes_dispatched"].append(row.get("regime_dispatched", "—"))
            if row["action"] in ("BUY", "SELL"):
                STATE["actions"].append((row["time"], float(row["price"]), row["action"]))
    print(f"[csv] loaded {len(STATE['times'])} rows from {CSV_PATH}")


def _append_csv_row(info, regime_raw, regime_dispatched):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "time": info["time"],
            "price": info["price"],
            "portfolio_value": info["portfolio_value"],
            "cash": info["cash"],
            "share_value": info["shares"] * info["price"],
            "exposure": info.get("exposure", 0.0),
            "regime_raw": regime_raw,
            "regime_dispatched": regime_dispatched,
            "action": info["action"],
        })


_load_state_from_csv()


# ---------- Alpaca clock helpers --------------------------------------
_TRADING_CLIENT = None


def _trading_client():
    global _TRADING_CLIENT
    if _TRADING_CLIENT is None:
        _TRADING_CLIENT = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=True)
    return _TRADING_CLIENT


def fetch_equity():
    try:
        acct = _trading_client().get_account()
        return float(getattr(acct, "equity", 0) or 0)
    except Exception as e:
        print(f"[fetch_equity] {e}")
        return None


def wait_for_open():
    while True:
        with LOCK:
            if STATE["stop_requested"]:
                return
        try:
            clock = _trading_client().get_clock()
        except Exception as e:
            with LOCK:
                STATE["status"] = f"clock error: {e}"
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


# ---------- meta-controller dispatch ----------------------------------
def _load_models_and_hmm():
    """Load all three specialists + the symbol's HMM."""
    models = {}
    for r in REGIMES:
        path = os.path.join(regime_model_dir(r), "best_model.zip")
        if not os.path.exists(path):
            raise RuntimeError(f"missing specialist: {path}")
        models[r] = SAC.load(path)
        print(f"[load] {r} ← {path}")

    hmm_path = os.path.join(HMM_MODELS_DIR, f"{SYMBOL}.pkl")
    if not os.path.exists(hmm_path):
        raise RuntimeError(f"missing HMM for {SYMBOL}: {hmm_path}")
    det = HMMRegimeDetector.load(hmm_path)
    print(f"[load] HMM {SYMBOL} ← {hmm_path}")
    return models, det


def _decide_regime(det, prices_all):
    """Return (raw_regime, dispatched_regime, gate_fired)."""
    labels = det.predict(prices_all)
    raw = str(labels[-1])
    dispatched = raw
    gate_fired = False
    if DRAWDOWN_GATE_THRESHOLD is not None and raw == "bear":
        dd = _drawdown_series(prices_all, DRAWDOWN_GATE_WINDOW)
        if float(dd[-1]) > DRAWDOWN_GATE_THRESHOLD:
            dispatched = "sideways"
            gate_fired = True
    return raw, dispatched, gate_fired


def trading_session(models, det):
    """One open→close trading session. Returns when the market closes."""
    if USE_ALPACA:
        live = AlpacaLiveMarket(symbol=SYMBOL, interval=INTERVAL)
    else:
        live = LiveMarket(symbol=SYMBOL, interval=INTERVAL,
                          initial_cash=INITIAL_CASH)
    obs = live.reset()

    with LOCK:
        STATE["status"] = f"trading ({INTERVAL})"

    while True:
        try:
            clock = _trading_client().get_clock()
        except Exception as e:
            with LOCK:
                STATE["status"] = f"clock error: {e}"
            time.sleep(POLL_SECONDS)
            continue

        with LOCK:
            if STATE["stop_requested"]:
                STATE["status"] = "stopping..."
                return
        if not getattr(clock, "is_open", False):
            with LOCK:
                STATE["status"] = "market closed"
            return

        try:
            raw_regime, disp_regime, gate_fired = _decide_regime(
                det, live._prices_all)
            agent = models[disp_regime]

            action, _ = agent.predict(obs, deterministic=True)
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
            STATE["cash"].append(max(0, info["cash"]))
            STATE["share_value"].append(max(0, info["shares"] * info["price"]))
            STATE["exposure"].append(info["exposure"])
            STATE["regimes_dispatched"].append(disp_regime)
            STATE["current_regime_raw"] = raw_regime
            STATE["current_regime_dispatched"] = disp_regime
            if gate_fired:
                STATE["gate_fires"] += 1
            if info["action"] in ("BUY", "SELL"):
                STATE["actions"].append((info["time"], info["price"], info["action"]))

        _append_csv_row(info, raw_regime, disp_regime)

        # Sleep until the next bar — but wake early if market closes or stop.
        bar_secs = LiveMarket._BAR_SECONDS.get(INTERVAL, 86400)
        target = time.time() + bar_secs + 5
        while time.time() < target:
            remaining = target - time.time()
            time.sleep(max(0.1, min(POLL_SECONDS, remaining)))
            try:
                if not getattr(_trading_client().get_clock(), "is_open", False):
                    break
            except Exception as e:
                print(f"[trading_session] clock poll error: {e}")
            with LOCK:
                if STATE["stop_requested"]:
                    break


def trading_loop():
    try:
        models, det = _load_models_and_hmm()
    except Exception as e:
        with LOCK:
            STATE["status"] = f"load error: {e}"
        print(f"[trading_loop] load error: {e}")
        return

    while True:
        with LOCK:
            if STATE["stop_requested"]:
                break
            STATE["status"] = "waiting for open"
        wait_for_open()

        with LOCK:
            if STATE["stop_requested"]:
                break

        try:
            trading_session(models, det)
        except Exception as e:
            with LOCK:
                STATE["status"] = f"session crashed: {e}"
            print(f"[trading_loop] session crashed: {e}")
            time.sleep(POLL_SECONDS)

    with LOCK:
        STATE["status"] = "stopped"
    print("[trading_loop] stopped.")


# ----------------------------- Dash app -------------------------------
app = dash.Dash(__name__)
app.title = f"{SYMBOL} V2 meta-controller"

app.layout = html.Div(
    style={"fontFamily": "monospace", "maxWidth": "1200px", "margin": "auto"},
    children=[
        html.H2(f"{SYMBOL} — V2 Regime-Switching SAC (paper)"),
        html.Div(
            "Server disconnected — restart deployment.py",
            id="disconnected-banner",
            style={"display": "none", "backgroundColor": "#ffcccc", "color": "#900",
                   "padding": "8px 12px", "marginBottom": "10px", "borderRadius": "4px",
                   "fontWeight": "bold"}),
        html.Div(
            [
                html.Div(id="status", style={"color": "#555", "flex": "1"}),
                html.Button(
                    "Stop Trading",
                    id="stop-btn",
                    n_clicks=0,
                    style={"backgroundColor": "#cc0000", "color": "white", "border": "none",
                           "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer",
                           "fontWeight": "bold", "fontFamily": "monospace"}),
                html.Span(id="stop-label",
                          style={"marginLeft": "10px", "color": "#888"}),
            ],
            style={"display": "flex", "alignItems": "center",
                   "gap": "15px", "marginBottom": "10px"}),
        html.Div(id="regime-panel", style={"marginBottom": "12px"}),
        dcc.Graph(id="chart"),
        dcc.Interval(id="tick", interval=5000, n_intervals=0),
        html.Div(id="_heartbeat", style={"display": "none"}),
    ],
)


@app.callback(Output("stop-label", "children"),
              Input("stop-btn", "n_clicks"),
              prevent_initial_call=True)
def stop_trading(_):
    with LOCK:
        STATE["stop_requested"] = True
    return "Stopping..."


@app.callback(
    [Output("chart", "figure"), Output("status", "children"),
     Output("regime-panel", "children"),
     Output("disconnected-banner", "style")],
    [Input("tick", "n_intervals")],
)
def update(_):
    with LOCK:
        times = list(STATE["times"])
        prices = list(STATE["prices"])
        cash = list(STATE["cash"])
        share_value = list(STATE["share_value"])
        exposure = list(STATE["exposure"])
        regimes = list(STATE["regimes_dispatched"])
        actions = list(STATE["actions"])
        status = STATE["status"]
        frozen = STATE["frozen_equity"]
        raw_regime = STATE["current_regime_raw"]
        disp_regime = STATE["current_regime_dispatched"]
        gate_fires = STATE["gate_fires"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.3, 0.15],
        subplot_titles=(f"{SYMBOL} price", "Portfolio breakdown ($)", "Dispatched regime"))

    # Row 1: price + trade markers
    fig.add_trace(go.Scatter(x=times, y=prices, mode="lines", name="price",
                             line=dict(color="#1f77b4")), row=1, col=1)
    for side, color, sym in [("BUY", "green", "triangle-up"),
                              ("SELL", "red", "triangle-down")]:
        pts = [(t, p) for (t, p, a) in actions if a == side]
        if pts:
            t, p = zip(*pts)
            fig.add_trace(go.Scatter(x=list(t), y=list(p), mode="markers", name=side,
                                     marker=dict(symbol=sym, color=color, size=12)),
                          row=1, col=1)

    # Row 2: portfolio stacked area
    if cash:
        fig.add_trace(go.Scatter(x=times, y=cash, mode="lines", name="cash",
                                 line=dict(width=0.5, color="#ff7f0e"),
                                 stackgroup="portfolio",
                                 fillcolor="rgba(255,127,14,0.4)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=times, y=share_value, mode="lines", name="share value",
                                 line=dict(width=0.5, color="#9467bd"),
                                 stackgroup="portfolio",
                                 fillcolor="rgba(148,103,189,0.4)"), row=2, col=1)
    elif frozen is not None:
        fig.add_trace(go.Scatter(x=[0, 1], y=[frozen, frozen], mode="lines",
                                 name=f"equity (frozen ${frozen:,.2f})",
                                 line=dict(color="#2ca02c", dash="dash")), row=2, col=1)

    # Row 3: regime strip
    if times and regimes:
        for r in REGIMES:
            mask = [1 if reg == r else None for reg in regimes]
            fig.add_trace(go.Scatter(x=times, y=mask, mode="markers", name=r,
                                     marker=dict(color=REGIME_COLORS[r], size=10,
                                                 symbol="square")),
                          row=3, col=1)
        fig.update_yaxes(range=[0.5, 1.5], showticklabels=False, row=3, col=1)

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(height=800, margin=dict(l=60, r=20, t=50, b=40),
                      showlegend=True, template="plotly_white",
                      uirevision="live-trading")

    # Regime panel
    current_exp = exposure[-1] if exposure else 0.0
    gate_note = (" (redirected to sideways)" if raw_regime == "bear"
                 and disp_regime == "sideways" else "")
    regime_panel = html.Div(
        [
            html.Span("HMM raw: ", style={"color": "#555"}),
            html.Span(raw_regime.upper(),
                      style={"color": REGIME_COLORS.get(raw_regime, "#000"),
                             "fontWeight": "bold", "marginRight": "20px"}),
            html.Span("Dispatched: ", style={"color": "#555"}),
            html.Span(disp_regime.upper() + gate_note,
                      style={"color": REGIME_COLORS.get(disp_regime, "#000"),
                             "fontWeight": "bold", "marginRight": "20px"}),
            html.Span(f"Exposure: {current_exp*100:.1f}%  |  ",
                      style={"color": "#555"}),
            html.Span(f"Gate fires: {gate_fires}", style={"color": "#555"}),
        ],
        style={"padding": "10px", "backgroundColor": "#f5f5f5",
               "borderRadius": "4px", "fontSize": "14px"})

    equity_str = f"  |  equity: ${frozen:,.2f}" if frozen is not None else ""
    status_text = (f"Status: {status}  |  interval: {INTERVAL}  |  "
                   f"bars: {len(times)}{equity_str}")

    return fig, status_text, regime_panel, {"display": "none"}


app.clientside_callback(
    """
    function(n) {
        window._lastUpdate = Date.now();
        if (!window._disconnectTimer) {
            window._disconnectTimer = setInterval(function() {
                var last = window._lastUpdate || Date.now();
                var banner = document.getElementById('disconnected-banner');
                if (banner) {
                    banner.style.display = (Date.now() - last > 15000) ? 'block' : 'none';
                }
            }, 3000);
        }
        return '';
    }
    """,
    Output("_heartbeat", "children"),
    Input("tick", "n_intervals"),
)


# --------------------------------- main -------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    app.run(debug=False, host="127.0.0.1", port=DASH_PORT)
