# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Train the BULL regime specialist.
Exposure target 0.95 — strong incentive to stay invested during uptrends.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import ensure_hmms_fitted, train_regime

ensure_hmms_fitted()

train_regime(
    regime="bull",
    exposure_target=0.95,
    timesteps=2_000_000,
    description="Bull agent: stay invested, ride momentum",
)
