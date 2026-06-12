"""Shared bootstrap for notebook style helpers.

Importing this module makes the repo root importable, silences noisy warnings,
and exposes thesis output directories for the few style helpers that can write
figures or tables.
"""

import logging
import os
from pathlib import Path
import sys
import warnings


def _find_repo_root():
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if ((candidate / ".logs").exists()
                and (candidate / "configs").exists()
                and (candidate / "_notebooks").exists()):
            return str(candidate)
    raise FileNotFoundError("Could not find repository root")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "explorer"))

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)

THESIS_ROOT = os.path.join(REPO_ROOT, "context", "MSc_Thesis")
FIG_DIR = os.path.join(THESIS_ROOT, "figures", "results")
TAB_DIR = os.path.join(THESIS_ROOT, "tables", "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
