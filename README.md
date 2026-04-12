# Machine Learning - DBUAS

Zentrales Repository fuer den Kurs Machine Learning an der DBUAS.

## Projektstruktur

```
machine_learning_DSM/
├── configs/
│   └── default.yaml                   # Zentrale Konfiguration (Pfade, Seeds, Plot-Style)
├── data/
│   ├── raw/                           # Rohdaten (Fake.csv, True.csv)
│   └── processed/                     # Generierte Artefakte
├── notebooks/
│   ├── 01_fake_news_supervised.ipynb  # Supervised Learning: Fake News Erkennung
│   ├── 02_unsupervised_learning.ipynb # Unsupervised Learning: Clustering & Dimensionsreduktion
│   └── 03_hotel_bookings_supervised.ipynb # Multiclass Supervised ML: Hotelbuchungen
├── src/
│   ├── data_loading.py                # Daten laden, Seed-Management
│   ├── plotting.py                    # Wiederverwendbare Plot-Funktionen
│   ├── text_features.py               # Text-Analyse Utilities
│   └── evaluation.py                  # Statistische Tests, Clustering-Metriken
└── requirements.txt                   # Python-Abhaengigkeiten
```

## Setup

```bash
pip install -r requirements.txt
```

## Notebooks

### 01 - Fake News Detection (Supervised Learning)
- Explorative Datenanalyse (EDA) des Fake/True News Datensatzes
- Text-Feature-Extraktion mit CountVectorizer (N-Grams, Wordclouds)
- Statistische Hypothesentests (paarweise T-Tests mit FDR-Korrektur)

### 03 - Hotel Bookings (Multiclass Supervised Learning)
- Zielvariable: `reservation_status` (Check-Out, Canceled, No-Show)
- EDA mit Klassenverteilung, Korrelationen, Feature-Verteilungen
- Feature Engineering (total_stays, total_guests, Country-Reduktion)
- Modellvergleich: Logistische Regression, Random Forest, Gradient Boosting
- Hyperparameter-Optimierung mit RandomizedSearchCV
- Feature Importance Analyse

### 02 - Unsupervised Learning & Dimensionsreduktion
- Online Retail: Hochdimensionale, sparse Daten
- K-Means Kundensegmentierung (Elbow-Methode, Silhouette Score)
- Clustering-Vergleich: K-Means vs. Agglomerativ vs. DBSCAN
- Dimensionsreduktion: PCA und t-SNE (Olivetti Faces)
- Autoencoder fuer Bildkompression (MNIST, TensorFlow/Keras)

## Konfiguration

Alle geteilten Einstellungen (Datenpfade, Random Seeds, Plot-Styles) liegen in
`configs/default.yaml` und werden in den Notebooks per `yaml.safe_load()` geladen.

## Daten

Rohdaten liegen in `data/raw/`. Generierte Outputs kommen nach `data/processed/`.
Die Unsupervised-Notebooks laden zusaetzliche Datensaetze direkt von externen URLs.
