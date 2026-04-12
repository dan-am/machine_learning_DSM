import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def build_count_vectorizer(texts, ngram_range=(1, 2), stop_words="english"):
    """Erstellt einen CountVectorizer und gibt (sparse_matrix, feature_names, vectorizer) zurueck."""
    cv = CountVectorizer(ngram_range=tuple(ngram_range), stop_words=stop_words)
    X = cv.fit_transform(texts)
    return X, cv.get_feature_names_out(), cv


def get_top_n_words(X_sparse, feature_names, labels, label_value, n=20):
    """Extrahiert die haeufigsten N-Grams fuer ein bestimmtes Label."""
    indices = [i for i, l in enumerate(labels) if l == label_value]
    if not indices:
        return pd.DataFrame()
    sums = X_sparse[indices].sum(axis=0)
    data = [(term, sums[0, col]) for col, term in enumerate(feature_names)]
    ranking = pd.DataFrame(data, columns=["term", "count"])
    return ranking.sort_values("count", ascending=False).head(n)
