import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import io
import base64
from io import BytesIO


def plot_elbow_curve(wcss):
    """Plot the elbow curve for WCSS values"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)

    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')



def plot_dendrogram(X, n_clusters):
    """Generate a dendrogram with a cut-off line for the specified number of clusters."""
    # Perform linkage for dendrogram
    linkage_matrix = linkage(X, method='ward', metric='euclidean')

    # Create dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendro = dendrogram(linkage_matrix, truncate_mode='level', p=5)

    # Calculate the cut-off height for n_clusters
    if n_clusters > 1:
        # Find the height where the dendrogram is cut to produce n_clusters
        cut_height = linkage_matrix[-(n_clusters - 1), 2]
        ax.axhline(y=cut_height, color='red', linestyle='--',
                   label=f'Cut-off for {n_clusters} clusters')

    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Sample Index or (Cluster Size)')
    ax.set_ylabel('Distance (Ward)')
    ax.legend()
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_cluster_distribution(labels):
    """Generate a bar plot for cluster distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique, counts, color='#007bff')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Cluster Distribution')
    plt.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')