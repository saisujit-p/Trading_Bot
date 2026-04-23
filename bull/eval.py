# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Evaluate the BULL specialist on OOS bull-only segments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import eval_regime

eval_regime(regime="bull", exposure_target=0.95)
