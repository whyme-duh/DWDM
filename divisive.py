import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Custom dataset
X = np.array([[1, 3], [4, 2], [2, 3],[6, 71], [8, 9], [10,12],[10,15],
              [50, 19], [20, 20], [44, 21],[46, 18], [51, 19],
              [50, 101], [40, 22], [50, 23], [49, 53], [50, 25],
              [50, 50], [50, 45], [44, 51],[46, 55], [51, 48],
              [50, 58], [40, 49], [50, 55],[49, 53], [50, 52],
              [25,80], [100, 30],[150, 90]])

# Perform Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=50, linkage='ward')
cluster.fit(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
plt.title('Divisive Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
