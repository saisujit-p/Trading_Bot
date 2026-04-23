# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Train the BEAR regime specialist.
Exposure target 0.20 — punishes being levered long during downtrends.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import ensure_hmms_fitted, train_regime

ensure_hmms_fitted()

train_regime(
    regime="bear",
    exposure_target=0.20,
    timesteps=1_000_000,
    description="Bear agent: defensive, preserve capital",
)
