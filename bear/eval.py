# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Evaluate the BEAR specialist on OOS bear-only segments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import eval_regime

eval_regime(regime="bear", exposure_target=0.20)
