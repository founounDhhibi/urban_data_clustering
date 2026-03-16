"""
Urban Data Clustering Project
==============================
Applied K-Means and PCA on a synthetic geospatial urban dataset.
Includes EDA, feature scaling, cluster evaluation, and a sklearn Pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

np.random.seed(42)

N_ZONES = 500
N_TRUE_CLUSTERS = 4

# Generate base cluster structure
X_base, true_labels = make_blobs(
    n_samples=N_ZONES,
    n_features=6,
    centers=N_TRUE_CLUSTERS,
    cluster_std=1.2,
    random_state=42
)

# Map to meaningful urban features
feature_names = [
    "population_density",   # people per km²
    "avg_income",           # average household income (k$)
    "crime_rate",           # crimes per 1000 residents
    "green_space_ratio",    # % of area as parks/nature
    "transit_accessibility",# 0–100 score
    "building_age_avg"      # average building age (years)
]

# Scale to realistic ranges
scales = [
    (500, 8000),   # population_density
    (20, 120),     # avg_income
    (1, 50),       # crime_rate
    (0, 60),       # green_space_ratio
    (10, 100),     # transit_accessibility
    (5, 80)        # building_age_avg
]

X_scaled = np.zeros_like(X_base)
for i, (lo, hi) in enumerate(scales):
    col = X_base[:, i]
    col_norm = (col - col.min()) / (col.max() - col.min())
    X_scaled[:, i] = col_norm * (hi - lo) + lo

# Add lat/lon columns for geo flavor
lat = np.random.uniform(36.7, 36.9, N_ZONES)   # Tunis-like coords
lon = np.random.uniform(10.1, 10.3, N_ZONES)

df = pd.DataFrame(X_scaled, columns=feature_names)
df["latitude"] = lat
df["longitude"] = lon
df["zone_id"] = [f"ZONE_{i:04d}" for i in range(N_ZONES)]

df.to_csv("data/urban_zones.csv", index=False)
print("✅ Dataset generated and saved to data/urban_zones.csv")
print(f"   Shape: {df.shape}\n")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

print("=" * 55)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 55)
print("\n📊 Summary Statistics:")
print(df[feature_names].describe().round(2).to_string())

# --- Distribution plots ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
palette = sns.color_palette("mako", N_TRUE_CLUSTERS + 2)

for ax, feat, color in zip(axes.flat, feature_names, palette[2:]):
    ax.hist(df[feat], bins=30, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: outputs/01_feature_distributions.png")

# --- Correlation heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[feature_names].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.5, ax=ax, square=True
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/02_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 3. FEATURE SCALING
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("3. FEATURE SCALING")
print("=" * 55)

X = df[feature_names].values
scaler = StandardScaler()
X_scaled_std = scaler.fit_transform(X)

print(f"   Mean after scaling  : {X_scaled_std.mean(axis=0).round(4)}")
print(f"   Std  after scaling  : {X_scaled_std.std(axis=0).round(4)}")
print("✅ Features standardized (mean=0, std=1)")

# ─────────────────────────────────────────────
# 4. ELBOW METHOD — OPTIMAL K
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("4. ELBOW METHOD")
print("=" * 55)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled_std)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled_std, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(K_range, inertias, "o-", color="#2196F3", linewidth=2.5, markersize=7)
ax1.axvline(x=4, color="#F44336", linestyle="--", alpha=0.7, label="Optimal k=4")
ax1.set_title("Elbow Method — Inertia vs K", fontsize=13, fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(K_range, silhouettes, "s-", color="#4CAF50", linewidth=2.5, markersize=7)
ax2.axvline(x=4, color="#F44336", linestyle="--", alpha=0.7, label="Optimal k=4")
ax2.set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle("Choosing the Optimal Number of Clusters", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/03_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/03_elbow_silhouette.png")
print(f"   Best silhouette score at k=4 : {silhouettes[2]:.4f}")

# ─────────────────────────────────────────────
# 5. K-MEANS CLUSTERING (k=4)
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("5. K-MEANS CLUSTERING  (k=4)")
print("=" * 55)

OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled_std)
df["cluster"] = cluster_labels

print("\n📊 Cluster sizes:")
print(df["cluster"].value_counts().sort_index().to_string())

print("\n📊 Cluster feature means (original scale):")
cluster_means = df.groupby("cluster")[feature_names].mean().round(2)
print(cluster_means.to_string())

# ─────────────────────────────────────────────
# 6. PCA VISUALIZATION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("6. PCA VISUALIZATION")
print("=" * 55)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_std)

explained = pca.explained_variance_ratio_
print(f"   PC1 explains : {explained[0]*100:.1f}%")
print(f"   PC2 explains : {explained[1]*100:.1f}%")
print(f"   Total        : {sum(explained)*100:.1f}%")

# Full PCA scatter
colors = ["#E53935", "#8E24AA", "#039BE5", "#43A047"]
cluster_names = {
    0: "High-Density Urban Core",
    1: "Suburban Residential",
    2: "Green & Low-Crime Zone",
    3: "Transit-Rich District"
}

fig, ax = plt.subplots(figsize=(10, 7))
for c in range(OPTIMAL_K):
    mask = cluster_labels == c
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[c], label=f"Cluster {c}: {cluster_names[c]}",
        alpha=0.7, s=40, edgecolors="white", linewidth=0.3
    )

# Plot centroids in PCA space
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1],
    c="black", marker="X", s=200, zorder=5, label="Centroids"
)

ax.set_title(
    f"K-Means Clusters in PCA Space\n"
    f"(PC1={explained[0]*100:.1f}%  PC2={explained[1]*100:.1f}%)",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel(f"Principal Component 1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"Principal Component 2 ({explained[1]*100:.1f}%)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("outputs/04_pca_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/04_pca_clusters.png")

# Geo scatter
fig, ax = plt.subplots(figsize=(9, 7))
for c in range(OPTIMAL_K):
    mask = cluster_labels == c
    ax.scatter(
        df.loc[mask, "longitude"], df.loc[mask, "latitude"],
        c=colors[c], label=f"Cluster {c}",
        alpha=0.7, s=30, edgecolors="white", linewidth=0.2
    )
ax.set_title("Urban Zone Clusters — Geospatial View", fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/05_geo_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: outputs/05_geo_clusters.png")

# ─────────────────────────────────────────────
# 7. CLUSTER EVALUATION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("7. CLUSTER EVALUATION")
print("=" * 55)

final_silhouette = silhouette_score(X_scaled_std, cluster_labels)
inertia = kmeans.inertia_

print(f"   Silhouette Score  : {final_silhouette:.4f}  (range -1 to 1, higher=better)")
print(f"   Inertia (WCSS)    : {inertia:.2f}")
print(f"   Num clusters      : {OPTIMAL_K}")

# Silhouette interpretation
if final_silhouette > 0.5:
    interpretation = "Strong cluster structure"
elif final_silhouette > 0.25:
    interpretation = "Reasonable cluster structure"
else:
    interpretation = "Weak cluster structure"
print(f"   Interpretation    : {interpretation}")

# ─────────────────────────────────────────────
# 8. SKLEARN PIPELINE
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("8. SKLEARN PIPELINE")
print("=" * 55)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=2, random_state=42)),
    ("kmeans", KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10))
])

pipeline_labels = pipeline.fit_predict(X)
pipeline_silhouette = silhouette_score(
    pipeline.named_steps["pca"].transform(
        pipeline.named_steps["scaler"].transform(X)
    ),
    pipeline_labels
)

print("   Pipeline steps:")
for name, step in pipeline.named_steps.items():
    print(f"     → {name}: {step.__class__.__name__}")
print(f"\n   Pipeline Silhouette Score: {pipeline_silhouette:.4f}")
print("✅ Pipeline built and evaluated successfully")

# ─────────────────────────────────────────────
# 9. SUMMARY REPORT
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  Dataset       : {N_ZONES} urban zones, {len(feature_names)} features")
print(f"  Algorithm     : K-Means (k={OPTIMAL_K})")
print(f"  Silhouette    : {final_silhouette:.4f}")
print(f"  PCA variance  : {sum(explained)*100:.1f}% (2 components)")
print(f"  Pipeline      : StandardScaler → PCA → KMeans")
print("\n  Output files saved in outputs/:")
for f in ["01_feature_distributions.png", "02_correlation_heatmap.png",
          "03_elbow_silhouette.png", "04_pca_clusters.png", "05_geo_clusters.png"]:
    print(f"    • {f}")
print("\n✅ Project complete!")
