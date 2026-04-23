# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""
Force-refit every per-symbol HMM and wipe the old cache first.

Use this any time the HMM feature schema changes (e.g., feature_window,
number of input features) or when you want a clean re-fit. Loading a
cached pkl that was fit on a different feature shape will error at predict
time, so wiping is the safe default.
"""
import os
import shutil

from env import HMM_MODELS_DIR
from _shared import fit_hmms


def main():
    if os.path.isdir(HMM_MODELS_DIR):
        print(f"[refit] wiping {HMM_MODELS_DIR}")
        shutil.rmtree(HMM_MODELS_DIR)
    os.makedirs(HMM_MODELS_DIR, exist_ok=True)

    fit_hmms(force=True)
    print("\n[refit] done. Run diagnose_hmm.py to validate.")


if __name__ == "__main__":
    main()
