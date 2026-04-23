# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
HMM-based market regime detection with directional features.

Fits a Gaussian Hidden Markov Model on a 4-dim feature vector per bar:
  [rolling_mean_return, rolling_std_return, trend_slope, drawdown]

  - rolling_mean / rolling_std: short-term return statistics (feature_window).
  - trend_slope: slope of log-price OLS over trend_window bars. Encodes
    directional drift independent of volatility.
  - drawdown: log(price / rolling_max_price) over drawdown_window bars.
    Zero means at the peak, very negative means deep in a drawdown.

State → regime labeling is driven by *trend_slope*: the state with the
most-negative mean slope is 'bear', most-positive is 'bull', middle is
'sideways'. This fixes the prior pitfall where pure vol-based labeling
called calm-but-positive-drift periods "sideways" and turbulent rallies
"bear" — now direction is the primary sort key.

Requires: pip install hmmlearn
"""
import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """
    Fit once on a reference price series, then decode any new prices.
    Always labels the most recent bar via forward probabilities (causal).
    """

    def __init__(self, n_states=3, covariance_type="diag", n_iter=200,
                 seed=42, feature_window=20, trend_window=60,
                 drawdown_window=252, stickiness=200.0, init_diag=0.97):
        """
        feature_window: window for rolling mean/std of log returns.
        trend_window: window for the log-price OLS slope — the primary
            directional signal. 60 bars ≈ 3 trading months for daily data.
        drawdown_window: window for the rolling max used to compute
            drawdown. 252 bars ≈ 1 trading year.
        stickiness: Dirichlet pseudocount added to the diagonal of the
            transition matrix prior. Higher = stronger bias toward staying
            in the current regime.
        init_diag: initial diagonal of transmat_ before EM starts.
        """
        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.feature_window = int(feature_window)
        self.trend_window = int(trend_window)
        self.drawdown_window = int(drawdown_window)
        self.stickiness = float(stickiness)
        self.init_diag = float(init_diag)
        self.model = None
        self.state_to_regime = {}

    # ---------- fit ----------
    def fit(self, prices, features=None):
        """
        Fit the HMM. Uses rolling-mean/rolling-std features by default.
        Applies a Dirichlet prior biased toward sticky self-transitions so
        EM doesn't converge to an oscillating regime sequence.
        """
        X = self._build_features(prices, features)

        n = self.n_states
        # Dirichlet prior: high diagonal, low off-diagonal pseudocounts.
        transmat_prior = np.ones((n, n), dtype=np.float64) + \
                         np.eye(n, dtype=np.float64) * self.stickiness

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.seed,
            transmat_prior=transmat_prior,
            init_params="smc",  # 't' omitted — we set transmat_ manually below
            params="stmc",
        )

        # Seed transmat_ with a sticky initialization so the first M-step
        # isn't biased toward the oscillating solution.
        off = (1.0 - self.init_diag) / max(n - 1, 1)
        transmat_init = np.full((n, n), off, dtype=np.float64)
        np.fill_diagonal(transmat_init, self.init_diag)
        self.model.transmat_ = transmat_init
        # startprob_ is still learned via init_params='s'; no need to set here.

        self.model.fit(X)

        # Directional labeling: rank states by mean trend_slope (feature idx 2).
        # The state with the most-negative slope is 'bear' (persistent decline),
        # most-positive is 'bull' (persistent advance), middle is 'sideways'.
        # This replaces the old risk-adjusted-return sort, which was dominated
        # by volatility and mis-labeled calm-uptrend periods as 'sideways'.
        trend_slopes = self.model.means_[:, 2]
        order = np.argsort(trend_slopes)  # ascending: most-negative → most-positive
        if self.n_states >= 3:
            self.state_to_regime[int(order[0])] = 'bear'
            self.state_to_regime[int(order[-1])] = 'bull'
            for s in order[1:-1]:
                self.state_to_regime[int(s)] = 'sideways'
        elif self.n_states == 2:
            self.state_to_regime[int(order[0])] = 'bear'
            self.state_to_regime[int(order[-1])] = 'bull'
        else:
            raise ValueError("n_states must be >= 2")
        return self

    # ---------- decode ----------
    def predict(self, prices, features=None):
        """
        Return per-bar regime labels. Output length matches len(prices).
        Uses Viterbi for the most likely regime sequence.
        """
        self._check_fit()
        X = self._build_features(prices, features)
        states = self.model.predict(X)  # Viterbi
        labels = np.array([self.state_to_regime[int(s)] for s in states], dtype='U8')
        # Returns/features have length N-1 (diff drops one); pad first bar.
        return np.concatenate([[labels[0]], labels])

    def predict_current(self, prices, features=None):
        """Return the regime label for the most recent bar (deployment use)."""
        return str(self.predict(prices, features)[-1])

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "seed": self.seed,
                "feature_window": self.feature_window,
                "trend_window": self.trend_window,
                "drawdown_window": self.drawdown_window,
                "stickiness": self.stickiness,
                "init_diag": self.init_diag,
                "model": self.model,
                "state_to_regime": self.state_to_regime,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Reject pkls fit with the old 2-feature schema — input dim mismatch
        # would silently decode garbage.
        if "trend_window" not in data:
            raise ValueError(
                f"{path}: legacy HMM pkl detected (missing trend_window / "
                "drawdown_window). Delete V2/SAC/hmm_models/ and refit — "
                "the feature schema changed."
            )
        det = cls(
            n_states=data["n_states"],
            covariance_type=data["covariance_type"],
            n_iter=data["n_iter"],
            seed=data["seed"],
            feature_window=data["feature_window"],
            trend_window=data["trend_window"],
            drawdown_window=data["drawdown_window"],
            stickiness=data.get("stickiness", 200.0),
            init_diag=data.get("init_diag", 0.97),
        )
        det.model = data["model"]
        det.state_to_regime = data["state_to_regime"]
        return det

    # ---------- info ----------
    def describe(self):
        """Print per-state mean of each of the 4 features."""
        self._check_fit()
        print(f"HMM regimes (n_states={self.n_states}, "
              f"feature_window={self.feature_window}, "
              f"trend_window={self.trend_window}, "
              f"drawdown_window={self.drawdown_window}):")
        for state, regime in sorted(self.state_to_regime.items()):
            m = self.model.means_[state]
            print(f"  state {state} → {regime:>8s}  "
                  f"roll_mean={m[0]*100:+.3f}%  "
                  f"roll_vol={m[1]*100:.3f}%  "
                  f"trend_slope={m[2]*100:+.4f}%/bar  "
                  f"drawdown={m[3]*100:+.2f}%")

    # ---------- helpers ----------
    def _build_features(self, prices, features=None):
        """
        Build HMM observations: 4 columns, each row causally aligned to bar i+1.
          0) rolling_mean of log returns over feature_window
          1) rolling_std of log returns over feature_window
          2) trend_slope: OLS slope of log_price over trend_window
          3) drawdown: log_price - max(log_price) over drawdown_window (<= 0)

        Expanding window is used for the first (W-1) bars of each window type.
        """
        prices = np.asarray(prices, dtype=np.float64)
        min_len = max(3, self.feature_window, self.trend_window)
        if len(prices) < min_len:
            raise ValueError(
                f"need at least {min_len} prices to fit/decode HMM "
                f"(feature_window={self.feature_window}, "
                f"trend_window={self.trend_window})"
            )
        log_prices = np.log(np.maximum(prices, 1e-12))
        log_returns = np.diff(log_prices)
        W = self.feature_window
        TW = self.trend_window
        DW = self.drawdown_window
        n = len(log_returns)  # == len(prices) - 1

        rolling_mean = np.empty(n, dtype=np.float64)
        rolling_std = np.empty(n, dtype=np.float64)
        trend_slope = np.empty(n, dtype=np.float64)
        drawdown = np.empty(n, dtype=np.float64)

        for i in range(n):
            # Rolling return statistics over the last W returns.
            lo = max(0, i - W + 1)
            chunk = log_returns[lo:i + 1]
            rolling_mean[i] = chunk.mean()
            rolling_std[i] = chunk.std() if len(chunk) > 1 else 1e-6

            # OLS slope of log_price over the last TW bars ending at i+1.
            t_lo = max(0, i + 2 - TW)
            y = log_prices[t_lo:i + 2]
            k = len(y)
            if k >= 2:
                t = np.arange(k, dtype=np.float64)
                t_mean = t.mean()
                y_mean = y.mean()
                denom = ((t - t_mean) ** 2).sum()
                trend_slope[i] = ((t - t_mean) * (y - y_mean)).sum() / max(denom, 1e-12)
            else:
                trend_slope[i] = 0.0

            # Drawdown from rolling max over DW bars (in log-price space).
            d_lo = max(0, i + 2 - DW)
            window_max = log_prices[d_lo:i + 2].max()
            drawdown[i] = log_prices[i + 1] - window_max

        X = np.column_stack([rolling_mean, rolling_std, trend_slope, drawdown])

        if features is not None:
            features = np.asarray(features, dtype=np.float64)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            if len(features) == len(prices):
                features = features[1:]  # align with diff
            if len(features) != n:
                raise ValueError("features length must match prices (or prices-1)")
            X = np.concatenate([X, features], axis=1)

        return X

    def _check_fit(self):
        if self.model is None:
            raise RuntimeError("HMM not fitted yet; call .fit(prices) first")


# ---------- convenience: carry-over split_by_regime from the SMA detector ----------
def split_by_regime(prices, features, labels, target_regime, min_length=30):
    """
    Re-export of SMA detector's segment splitter — same semantics, works with
    any labels array regardless of source (SMA or HMM).
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
