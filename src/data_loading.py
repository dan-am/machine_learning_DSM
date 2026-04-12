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


def load_fake_news_data(cfg):
    """Laedt Fake/True News CSVs, fuegt Labels hinzu und gibt einen gemischten DataFrame zurueck."""
    raw = Path(cfg["paths"]["raw_data"])
    fake = pd.read_csv(raw / cfg["fake_news"]["fake_file"])
    true = pd.read_csv(raw / cfg["fake_news"]["true_file"])
    fake["label"], true["label"] = 0, 1
    df = pd.concat([fake, true]).sample(frac=1, random_state=cfg["random_seed"]).reset_index(drop=True)
    return df
