# utils/clustering_utils.py


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np


def apply_kmeans(matrix, n_clusters=3):
   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
   return kmeans.fit_predict(matrix)


def apply_dbscan(matrix, eps=0.5, min_samples=2):
   dbscan = DBSCAN(eps=eps, min_samples=min_samples)
   return dbscan.fit_predict(matrix)


def apply_hierarchical(matrix, n_clusters=3):
   model = AgglomerativeClustering(n_clusters=n_clusters)
   return model.fit_predict(matrix)


def apply_pca(matrix, n_components=2):
   pca = PCA(n_components=n_components)
   components = pca.fit_transform(matrix)
   explained_variance = pca.explained_variance_ratio_
   return components, explained_variance


def apply_tsne(matrix, n_components=2, perplexity=30, n_iter=1000):
   # t-SNE requires dense input
   if hasattr(matrix, 'toarray'):
       matrix = matrix.toarray()
   matrix = StandardScaler().fit_transform(matrix)
   tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
   components = tsne.fit_transform(matrix)
   return components
