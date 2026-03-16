
# 🏙️ Urban Data Clustering Project

> Applied K-Means and PCA on a synthetic geospatial dataset to identify urban zone archetypes.

---

## 📌 Overview

This project clusters **500 synthetic urban zones** based on socio-economic and geographic features using unsupervised machine learning. The goal is to identify distinct neighborhood archetypes that can inform urban planning decisions.

## 🧠 Key Concepts

| Concept | Description |
|---------|-------------|
| **K-Means** | Partitions zones into K groups by minimizing within-cluster variance |
| **PCA** | Reduces 6 features to 2D for visualization while preserving variance |
| **Feature Scaling** | Standardizes features so none dominates the distance metric |
| **Silhouette Score** | Measures how well-separated the clusters are (range: -1 to 1) |
| **Elbow Method** | Finds optimal K by plotting inertia vs. number of clusters |

## 📁 Project Structure

```
urban_data_clustering/
│
├── urban_clustering.py        # Main Python script (runs end-to-end)
├── urban_clustering.ipynb     # Jupyter Notebook with explanations
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   └── urban_zones.csv        # Generated after running the script
│
└── outputs/
    ├── 01_feature_distributions.png
    ├── 02_correlation_heatmap.png
    ├── 03_elbow_silhouette.png
    ├── 04_pca_clusters.png
    └── 05_geo_clusters.png
```

## 🗃️ Dataset Features

| Feature | Description | Range |
|---------|-------------|-------|
| `population_density` | People per km² | 500–8000 |
| `avg_income` | Avg household income (k$) | 20–120 |
| `crime_rate` | Crimes per 1000 residents | 1–50 |
| `green_space_ratio` | % area as parks/nature | 0–60% |
| `transit_accessibility` | Transit score | 10–100 |
| `building_age_avg` | Average building age (years) | 5–80 |

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main script
python urban_clustering.py

# 3. Or open the notebook
jupyter notebook urban_clustering.ipynb
```

## 📊 Results

- **Optimal K = 4** clusters identified via Elbow Method + Silhouette Score
- **Silhouette Score ≈ 0.55** → Strong cluster separation
- **PCA explains ~75%** of total variance with 2 components

### Cluster Archetypes

| Cluster | Name | Key Traits |
|---------|------|-----------|
| 0 | High-Density Urban Core | High density, high income, good transit |
| 1 | Suburban Residential | Low density, medium income, older buildings |
| 2 | Green & Low-Crime Zone | High green space, low crime, lower density |
| 3 | Transit-Rich District | Excellent transit, moderate density |

## 🔧 Pipeline

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=4))
])
```

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn** — KMeans, PCA, Pipeline, StandardScaler
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization

---

*Synthetic dataset generated with `sklearn.datasets.make_blobs` — coordinates inspired by Tunis, Tunisia.*

