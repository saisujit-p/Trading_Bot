# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
HMM sanity-check: verify that labels align with what each regime *should* look like.

For each symbol:
  - Load the persisted HMM, label the full price series.
  - Report per-regime count, share, mean log-return, vol, avg run length.
  - Report transition matrix (regime persistence).
  - Save a PNG: price line colored by regime (green=bull, red=bear, gray=sideways).

Regime is healthy if:
  - bull mean return > sideways > bear
  - bear vol >= bull vol (typical)
  - avg run length >> 1 bar (stable labels, not flipping each bar)
"""
import os, sys
import numpy as np

from env import HMM_MODELS_DIR
from genObs import generate_observations
from _shared import SYMBOLS, PERIOD, INTERVAL

from hmm_regime import HMMRegimeDetector

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[warn] matplotlib not available — skipping plots")

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmm_diagnostics")
REGIME_COLORS = {"bull": "#2ecc71", "bear": "#e74c3c", "sideways": "#95a5a6"}


def _run_lengths(labels):
    """Return list of (regime, run_length) for each contiguous run."""
    runs = []
    if len(labels) == 0:
        return runs
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((str(labels[start]), i - start))
            start = i
    runs.append((str(labels[start]), len(labels) - start))
    return runs


def _transition_matrix(labels, regimes=("bear", "sideways", "bull")):
    """Row-normalized P(next | current). Counts only between-bar transitions."""
    idx = {r: i for i, r in enumerate(regimes)}
    M = np.zeros((len(regimes), len(regimes)), dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return M / row_sums


def diagnose(symbol):
    pkl = os.path.join(HMM_MODELS_DIR, f"{symbol}.pkl")
    if not os.path.exists(pkl):
        print(f"[skip] {symbol}: no HMM at {pkl}")
        return None

    _, full_df = generate_observations(symbol, PERIOD, INTERVAL)
    prices = full_df['close'].to_numpy(dtype=np.float64)

    det = HMMRegimeDetector.load(pkl)
    labels = det.predict(prices)

    log_rets = np.diff(np.log(prices))
    label_rets = labels[1:]  # align with log_rets (drops first bar)

    print(f"\n===== {symbol} (N={len(prices)} bars) =====")
    print("HMM learned states:")
    det.describe()

    regimes = ["bear", "sideways", "bull"]
    total = len(label_rets)
    print(f"\nPer-regime stats on full series:")
    print(f"  {'regime':>8s}  {'count':>6s}  {'share':>6s}  "
          f"{'mean_ret':>10s}  {'vol':>8s}  {'avg_run':>8s}")
    runs = _run_lengths(labels)

    for r in regimes:
        mask = (label_rets == r)
        n = int(mask.sum())
        if n == 0:
            print(f"  {r:>8s}  {'0':>6s}  {'-':>6s}  {'-':>10s}  {'-':>8s}  {'-':>8s}")
            continue
        rets = log_rets[mask]
        mean_ret = float(rets.mean())
        vol = float(rets.std())
        r_runs = [ln for reg, ln in runs if reg == r]
        avg_run = float(np.mean(r_runs)) if r_runs else 0.0
        print(f"  {r:>8s}  {n:>6d}  {100*n/total:>5.1f}%  "
              f"{mean_ret*100:>+9.3f}%  {vol*100:>7.3f}%  {avg_run:>8.1f}")

    T = _transition_matrix(labels, regimes)
    print(f"\nTransition matrix  (rows=from, cols=to):")
    print(f"  {'':>8s}  " + "  ".join(f"{r:>8s}" for r in regimes))
    for i, r in enumerate(regimes):
        row = "  ".join(f"{T[i,j]:>8.3f}" for j in range(len(regimes)))
        print(f"  {r:>8s}  {row}")
    # Diagonal = "stickiness" — higher is better (stable labels)
    diag = np.diag(T)
    print(f"  stickiness (diag avg): {diag.mean():.3f}  "
          f"(>0.9 = very stable, <0.7 = noisy)")

    if HAS_PLT:
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(prices))
        for r in regimes:
            mask = (labels == r)
            ax.scatter(x[mask], prices[mask], s=2,
                       c=REGIME_COLORS[r], label=r, alpha=0.8)
        ax.plot(x, prices, color="black", alpha=0.15, linewidth=0.5)
        ax.set_title(f"{symbol} — HMM regime labels (full {PERIOD} @ {INTERVAL})")
        ax.set_xlabel("bar"); ax.set_ylabel("price")
        ax.legend(markerscale=4, loc="upper left")
        out = os.path.join(PLOT_DIR, f"{symbol}.png")
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot: {out}")

    return {"symbol": symbol, "T": T, "labels": labels, "prices": prices}


def summarize(results):
    """Aggregate checks across all symbols."""
    valid = [r for r in results if r is not None]
    if not valid:
        return
    print("\n" + "=" * 60)
    print("  AGGREGATE HEALTH CHECKS")
    print("=" * 60)
    stickies = [float(np.diag(r["T"]).mean()) for r in valid]
    print(f"  mean stickiness across symbols: {np.mean(stickies):.3f}")
    worst = min(valid, key=lambda r: np.diag(r["T"]).mean())
    print(f"  least sticky: {worst['symbol']} ({np.diag(worst['T']).mean():.3f})")
    # Labeling sanity: risk-adjusted (mean - 0.5*vol) should be higher for bull
    # than for bear. Raw-mean comparison would flag short high-vol spike states
    # as bull, but we deliberately label those as bear.
    bad = 0
    starved = []
    for r in valid:
        labels, prices = r["labels"], r["prices"]
        log_rets = np.diff(np.log(prices))
        label_rets = labels[1:]

        def score(name):
            mask = (label_rets == name)
            if mask.sum() < 2:
                return None
            rets = log_rets[mask]
            return float(rets.mean() - 0.5 * rets.std())

        bull_s = score("bull")
        bear_s = score("bear")
        if bull_s is not None and bear_s is not None and bull_s <= bear_s:
            bad += 1
            print(f"  [WARN] {r['symbol']}: bull risk-adj score ({bull_s*100:+.3f}%) "
                  f"<= bear ({bear_s*100:+.3f}%)")

        total = len(label_rets)
        for reg in ("bull", "bear", "sideways"):
            share = 100 * (label_rets == reg).sum() / total
            if share < 3.0:
                starved.append((r["symbol"], reg, share))

    print(f"  mislabeled symbols (bull risk-adj <= bear): {bad}/{len(valid)}")
    if starved:
        print(f"\n  Regimes with <3% of bars (specialist will starve for training data):")
        for sym, reg, share in starved:
            print(f"    {sym:>6s}  {reg:>8s}  {share:>5.1f}%")


if __name__ == "__main__":
    results = [diagnose(s) for s in SYMBOLS]
    summarize(results)
