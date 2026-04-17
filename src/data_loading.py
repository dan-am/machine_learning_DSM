import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seeds(seed):
    """Setzt Random Seeds fuer Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path_str):
    """Loest relative Pfade gegen den Projekt-Root auf, absolute bleiben unveraendert."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_fake_news_data(cfg):
    """Laedt Fake/True News CSVs, fuegt Labels hinzu und gibt einen gemischten DataFrame zurueck."""
    raw = _resolve_path(cfg["paths"]["raw_data"])
    fake = pd.read_csv(raw / cfg["fake_news"]["fake_file"])
    true = pd.read_csv(raw / cfg["fake_news"]["true_file"])
    fake["label"], true["label"] = 0, 1
    df = pd.concat([fake, true]).sample(frac=1, random_state=cfg["random_seed"]).reset_index(drop=True)
    return df
