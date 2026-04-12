import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from statsmodels.stats.multitest import multipletests


def perform_multiple_ttests_with_correction(df, column, group_col, correction_method="fdr_bh"):
    """Fuehrt paarweise T-Tests mit Korrektur fuer multiples Testen durch.

    Args:
        df: DataFrame
        column: Spalte fuer den Vergleich (z.B. 'title_length')
        group_col: Gruppierungsspalte (z.B. 'subject')
        correction_method: Methode fuer p-Wert-Korrektur (default: FDR Benjamini-Hochberg)
    """
    groups = df[group_col].unique()
    pairs, t_stats, p_values = [], [], []

    for s1, s2 in itertools.combinations(groups, 2):
        data1 = df[df[group_col] == s1][column]
        data2 = df[df[group_col] == s2][column]
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
        pairs.append(f"{s1} vs {s2}")
        t_stats.append(t_stat)
        p_values.append(p_val)

    reject, p_corrected, _, _ = multipletests(p_values, method=correction_method)

    results_df = pd.DataFrame({
        "Vergleich": pairs,
        "T-Wert": t_stats,
        "p-Wert": p_values,
        "p-Wert (korrigiert)": p_corrected,
        "Signifikant": reject,
    })
    return results_df.sort_values("p-Wert (korrigiert)")


def evaluate_clustering(X, cluster_labels):
    """Berechnet Clustering-Metriken (Silhouette, Calinski-Harabasz, Davies-Bouldin)."""
    unique = np.unique(cluster_labels)
    if len(unique) > 1 and -1 not in unique:
        return {
            "Silhouette Score": silhouette_score(X, cluster_labels),
            "Calinski-Harabasz Score": calinski_harabasz_score(X, cluster_labels),
            "Davies-Bouldin Score": davies_bouldin_score(X, cluster_labels),
        }
    return {
        "Silhouette Score": np.nan,
        "Calinski-Harabasz Score": np.nan,
        "Davies-Bouldin Score": np.nan,
    }
