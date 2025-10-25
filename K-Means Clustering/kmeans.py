import random
import numpy as np

class KMeans:
  def __init__(self,n_clusters = 2,max_iter = 100):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.centroids = None

  def fit_predict(self,X):

    random_index = random.sample(range(0,X.shape[0]),self.n_clusters)
    self.centroids = X[random_index]

    for i in range(self.max_iter):
      # assigning clusters
      cluster_group = self.assign_clusters(X)

      # move centroids
      old_centroids = self.centroids
      self.centroids = self.move_centroids(X,cluster_group)


      #check finish
      if (old_centroids == self.centroids).all():
        break

    return cluster_group
  
  def assign_clusters(self,X):
    # vectorized approach to make it faster
    # Compute squared distances between all points and all centroids
    distances_sq = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :])**2, axis=2)

    # Assign each point to the closest centroid
    cluster_group = np.argmin(distances_sq, axis=1)
    return np.array(cluster_group)
  
  def move_centroids(self,X,cluster_group):
    new_centroids = []
    cluster_type = np.unique(cluster_group)

    for type in cluster_type:
      new_centroids.append(X[cluster_group == type].mean(axis = 0))


    return np.array(new_centroids)
  
  def calculate_wcss(self,X,cluster_group):
    wcss = 0
    for i in range(len(X)):
      cluster_id = cluster_group[i]
      centroid = self.centroids[cluster_id]
      wcss += np.sum((X[i] - centroid)**2)
    return wcss