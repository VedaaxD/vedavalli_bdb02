import numpy as np
import random

def euclidean(point,data):
    return np.sqrt(np.sum((point-data)**2,axis=1))

class KMeans:
    def __init__(self,n_clusters=3,max_iter=100):
        self.k=n_clusters
        self.max_iter=max_iter
    def fit(self,X):
        #initializing centroids using KMeans
        self.centroids=[X[random.randint(0,len(X)-1)]] #picks the first centroid randomly from the datapoints
        #self.centroids stores all the centroids (one per cluster)
        for _ in range(self.k-1):
            dists=np.min([euclidean(c,X) for c in self.centroids],axis=0) #list of arrays (each array is dist from one centroid c to every point in X)
            probs=dists/np.sum(dists)
            chosen=np.random.choice(len(X),p=probs)
            self.centroids.append(X[chosen])
        self.centroids=np.array(self.centroids)

        for _ in range(self.max_iter): #running the loop until covergence or max_iter
            clusters=[ [] for _ in range(self.k)]

            for x in X:
                idx=np.argmin(euclidean(x,self.centroids))
                clusters[idx].append(x)
            new_centroids=np.array([np.mean(cluster,axis=0) if cluster else self.centroids[i] for i,cluster in enumerate(clusters)] )

            if np.allclose(self.centroids,new_centroids):
                break
            self.centroids=new_centroids
    def predict(self,X):
        return [np.argmin(euclidean(x,self.centroids)) for x in X]
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)

print("Final centroids:\n", kmeans.centroids)
print("\nCluster labels for first 10 points:\n", labels[:10])

import matplotlib.pyplot as plt

# Convert labels to a NumPy array if needed
labels = np.array(labels)

# Plot the clusters
plt.figure(figsize=(8, 6))
for i in range(kmeans.k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

# Plot centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', s=200, marker='X', label='Centroids')

plt.title("K-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
