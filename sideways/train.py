# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Train the SIDEWAYS regime specialist.
Exposure target 0.80 — directional HMM-sideways is "trend-slope-near-zero,"
which on tech-bull tape means slow positive drift, not flat chop. Treating
it as a lite-bull (80% exposure) instead of a hedge (50%) recovers the
upside that was being forfeited on weakly-trending bars.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import ensure_hmms_fitted, train_regime

ensure_hmms_fitted()

train_regime(
    regime="sideways",
    exposure_target=0.80,
    timesteps=1_000_000,
    description="Sideways agent: lite-bull, ride mild drift",
)
