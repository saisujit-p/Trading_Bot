# Regime-Aware SAC Trading Bot

A reinforcement-learning trading system for US equities that routes every bar
through one of three specialist SAC agents — **bull**, **bear**, or
**sideways** — based on a per-symbol Hidden Markov Model regime detector.
The specialists are trained separately on regime-filtered segments with
distinct exposure targets and reward shaping, then composed at inference time
by a meta-controller.

Data is daily bars from yfinance for 15 large-cap US tech equities over a
10-year window. Evaluation is walk-forward OOS (last 20% of each series).

## How it works

**Regime detection (HMM).** One Gaussian HMM per symbol, fitted on a 4-dim
feature vector per bar:
`[rolling_mean_return, rolling_std_return, trend_slope, drawdown]`.
States are labeled by mean `trend_slope` — most-negative → *bear*,
most-positive → *bull*, middle → *sideways*. A Dirichlet prior on the
transition matrix biases the fit toward sticky self-transitions so EM doesn't
converge to an oscillating state sequence. See [hmm_regime.py](hmm_regime.py).

**Specialists.** Three SAC agents with shared observation space but different
reward shaping:

| Regime   | Exposure target | tx_cost_weight | Role                      |
|----------|-----------------|----------------|---------------------------|
| bull     | 0.95            | 1.0            | Stay invested, ride trends |
| bear     | 0.20            | 2.0            | Defensive, preserve capital |
| sideways | 0.80            | 1.0            | Lite-bull, ride mild drift |

Each specialist is trained only on contiguous regime-labeled segments
aggregated across all symbols, using `stable-baselines3`'s SAC with a
`MultiInputPolicy` and early stopping on no-improvement eval reward.

**Meta-controller.** At each OOS bar the HMM labels the regime, an optional
drawdown gate can redirect `bear → sideways` on shallow pullbacks, and the
selected specialist's deterministic action drives the env. See
[eval_meta.py](eval_meta.py).

## Layout

```
.
├── _shared.py          global config (symbols, period, reward shaping), training loop
├── data.py             yfinance data loader
├── env.py              Gymnasium wrapper over SimpleTradingEnv w/ regime filtering
├── tradingsys.py       bar-level trading simulator (cash, shares, fills, reward)
├── genObs.py           TA feature generation (RSI/MACD/ADX/EMA/BB/OBV + rolling stats)
├── hmm_regime.py       Gaussian HMM regime detector
├── regime_detector.py  legacy SMA-based detector (used as a baseline)
├── eval_meta.py        meta-controller OOS evaluation + per-regime attribution
├── training.py         end-to-end pipeline: fit HMMs → train 3 specialists → eval
├── overnight_training.py  sweeps hyperparameters across many nights of training
├── refit_hmms.py       forced HMM refit (e.g. after changing feature windows)
├── diagnose_hmm.py     per-symbol regime plots for sanity-checking the HMM
├── deployment.py       live paper-trading dashboard (Dash + Alpaca)
├── deployment_env.py   live-market adapter mimicking the backtest env API
├── bull/  bear/  sideways/
│                      per-specialist train.py and eval.py (call into _shared.py)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

CUDA-capable GPU recommended for SAC training (SB3 defaults to `device="cuda"`
in [_shared.py](_shared.py)). Set `device="cpu"` on `model = SAC(...)` if you
don't have one — training takes ~8× longer.

For live deployment with Alpaca, create `creds.txt` in this directory (excluded
by `.gitignore`):

```
API_KEY: <your_alpaca_api_key>
Secret:  <your_alpaca_secret>
```

## Running

**Full pipeline** (fit HMMs → train all 3 specialists → meta-controller OOS
eval). Expect ~several hours on a single GPU.

```bash
python training.py
```

**Train a single specialist** (after HMMs are fitted once):

```bash
python bull/train.py
python bear/train.py
python sideways/train.py
```

**Evaluate an individual specialist on its regime OOS segments:**

```bash
python bull/eval.py
```

**Full meta-controller OOS eval** (requires all three `best_model.zip` files
present under `bull/`, `bear/`, `sideways/`):

```bash
python eval_meta.py
```

**Live paper-trading dashboard** (connects to Alpaca paper account, visualizes
regime dispatch + portfolio on localhost):

```bash
python deployment.py
```

## Outputs

All trained artifacts are git-ignored. After a full `training.py` run you'll
have:

- `hmm_models/{SYMBOL}.pkl` — one HMM per symbol
- `bull/best_model.zip`, `bear/best_model.zip`, `sideways/best_model.zip` —
  best-by-eval SAC checkpoints
- `bull/logs/`, etc. — tensorboard events and `evaluations.npz`
- `trade_log*.csv` — live deployment trade logs (if running `deployment.py`)

## Config

Everything important lives at the top of [_shared.py](_shared.py):
`SYMBOLS`, `PERIOD`, `INTERVAL`, `TRAIN_FRAC`, `REGIME_METHOD`, and the
`REGIME_REWARD` shaping table. Change those and rerun the pipeline.

## License

Personal research project. No warranty, no trading advice.
