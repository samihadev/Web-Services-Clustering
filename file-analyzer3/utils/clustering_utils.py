import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def calculate_elbow_scores(X: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Calculate KMeans inertia scores for different numbers of clusters.

    Args:
        X: Input feature matrix
        max_clusters: Maximum number of clusters to evaluate

    Returns:
        Array of inertia scores for k=1 to k=max_clusters
    """
    inertias = np.zeros(max_clusters)
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias[k - 1] = kmeans.inertia_
    return inertias


def find_optimal_clusters(inertias: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the optimal number of clusters using an improved elbow method.

    Uses the kneedle algorithm approach to find the point of maximum curvature.

    Args:
        inertias: Array of inertia values from calculate_elbow_scores()

    Returns:
        Tuple of (optimal number of clusters, array of inertias)
    """
    # Normalize the inertias
    normalized = (inertias - inertias.min()) / (inertias.max() - inertias.min())

    # Calculate the difference from the straight line (perfect linear decrease)
    line = np.linspace(0, 1, len(inertias))
    differences = line - normalized

    # Find the point of maximum difference (elbow point)
    elbow_point = np.argmax(differences) + 1  # +1 because we start from k=1

    # Ensure we don't return the last point as the elbow
    if elbow_point >= len(inertias) - 1:
        elbow_point = max(2, len(inertias) - 2)  # At least 2 clusters

    return elbow_point, inertias


def calculate_hierarchical_wcss(X: np.ndarray, max_clusters: int = 10) -> np.ndarray:
    """
    Calculate WCSS for hierarchical clustering with different numbers of clusters.

    Args:
        X: Input feature matrix
        max_clusters: Maximum number of clusters to evaluate

    Returns:
        Array of WCSS values for k=1 to k=max_clusters
    """
    wcss = np.zeros(max_clusters)
    for k in range(1, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
        labels = model.fit_predict(X)

        # Calculate WCSS (sum of squared distances to cluster centers)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                wcss[k - 1] += np.sum((cluster_points - center) ** 2)

    return wcss


def apply_kmeans(X, n_clusters=12):
   model = KMeans(n_clusters=n_clusters, random_state=42)
   model.fit(X)
   return model.labels_


def apply_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> np.ndarray:
    """
    Apply DBSCAN clustering with robust noise handling and automatic parameter tuning.

    Args:
        X: Input feature matrix
        eps: Starting epsilon value (will be adjusted if needed)
        min_samples: Minimum samples parameter

    Returns:
        Array of cluster labels (guaranteed no noise points)
    """

    def _assign_noise_to_nearest_cluster(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Helper function to assign noise points to nearest clusters"""
        noise_indices = np.where(labels == -1)[0]
        clustered_indices = np.where(labels != -1)[0]

        if len(clustered_indices) == 0:
            return np.zeros_like(labels)  # All noise case

        nbrs = NearestNeighbors(n_neighbors=1).fit(X[clustered_indices])
        _, indices = nbrs.kneighbors(X[noise_indices])
        labels[noise_indices] = labels[clustered_indices[indices.flatten()]]
        return labels

    # Initial DBSCAN attempt
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    # If noise points exist, try to adjust parameters
    if -1 in labels:
        # Try increasing eps in geometric progression
        for multiplier in [1.1, 1.2, 1.3, 1.5, 2.0]:
            model = DBSCAN(eps=eps * multiplier, min_samples=min_samples)
            new_labels = model.fit_predict(X)
            if -1 not in new_labels:
                return new_labels

        # Try reducing min_samples if possible
        if min_samples > 2:
            for new_min_samples in [min_samples - 1, min_samples - 2, 2]:
                model = DBSCAN(eps=eps, min_samples=new_min_samples)
                new_labels = model.fit_predict(X)
                if -1 not in new_labels:
                    return new_labels

        # Final fallback - assign noise to nearest cluster
        labels = _assign_noise_to_nearest_cluster(X, labels)

    return labels


def apply_hierarchical(
        X: np.ndarray,
        n_clusters: int = None,
        linkage: str = 'ward'
) -> Tuple[np.ndarray, int]:
    """
    Apply hierarchical clustering with automatic cluster determination.

    Args:
        X: Input feature matrix
        n_clusters: Number of clusters (if None, determined automatically)
        linkage: Which linkage criterion to use ('ward', 'complete', 'average', 'single')

    Returns:
        Tuple of (cluster labels, number of clusters)
    """
    if n_clusters is None:
        max_clusters = min(15, len(X) - 1)
        wcss = calculate_hierarchical_wcss(X, max_clusters)
        n_clusters, _ = find_optimal_clusters(wcss)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',
        linkage=linkage
    )
    return model.fit_predict(X), n_clusters


def calculate_cluster_metrics(
        X: np.ndarray,
        labels: np.ndarray
) -> Dict[str, Union[float, Dict[int, int]]]:
    """
    Calculate various clustering quality metrics.

    Args:
        X: Input feature matrix
        labels: Cluster assignments

    Returns:
        Dictionary containing clustering metrics
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate cluster sizes
    cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))

    # Calculate WCSS (Within-Cluster Sum of Squares)
    wscc = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            cluster_center = cluster_points.mean(axis=0)
            wscc += np.sum((cluster_points - cluster_center) ** 2)

    # Calculate quality metrics (only if meaningful number of clusters)
    metrics = {
        'sizes': cluster_sizes,
        'wscc': wscc,
        'n_clusters': n_clusters
    }

    if 1 < n_clusters < len(X):
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    else:
        metrics['silhouette'] = None
        metrics['calinski_harabasz'] = None

    return metrics
def remove_duplicates(df):
   """Remove duplicate web services based on Input, Output, and Domaine columns"""
   # Create a hash for each row based on the key columns
   df['hash'] = df.apply(lambda row: hash(
       (str(row['Input']), str(row['output']), str(row['Domaine']))
   ), axis=1)


   # Keep only the first occurrence of each duplicate
   dedup_df = df.drop_duplicates(subset=['hash'])


   # Clean up
   dedup_df = dedup_df.drop(columns=['hash'])


   return dedup_df




from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from io import BytesIO
import base64