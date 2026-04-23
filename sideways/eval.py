# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Evaluate the SIDEWAYS specialist on OOS sideways-only segments."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared import eval_regime

eval_regime(regime="sideways", exposure_target=0.80)
