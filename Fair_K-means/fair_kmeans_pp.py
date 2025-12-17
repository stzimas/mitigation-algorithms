import numpy as np
from sklearn.metrics import silhouette_score


class KMeansBalanced_pp:
    def __init__(self, n_clusters=3, max_iter=400, random_state=None, lambda_=0.5, init_mode="random"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.lambda_ = lambda_
        self.centroids = None
        self.labels_ = None
        self.cluster_attributes = None
        self.attributes = None 
        self.stop_flag = None
        self.iter = 0
        self.init_mode = init_mode

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        if self.init_mode == "random":
            random_idx = np.random.permutation(X.shape[0])
            return X[random_idx[:self.n_clusters]]
        elif self.init_mode == "kmeans++":
            n_samples = X.shape[0]
            first_idx = np.random.randint(0, n_samples)
            centroids = [X[first_idx]]
            for _ in range(1, self.n_clusters):
                distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(n_samples, p=probabilities)
                centroids.append(X[next_idx])
            return np.array(centroids)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def find_clusters_r_b(self, attributes, labels):
        self.cluster_attributes = {}
        for i in range(self.n_clusters):
            cluster_mask = (labels == i)
            count_0 = np.sum(attributes[cluster_mask] == 0)
            count_1 = np.sum(attributes[cluster_mask] == 1)
            
            self.cluster_attributes[i] = {
                '0': count_0,
                '1': count_1,
                'majority': 0 if count_0 > count_1 else 1
            }

    def check_node_in_cluster(self, node, cluster, adj_0, adj_1):
        #if the node is in the cluster remove it

        if self.labels_ is not None and self.labels_[node] == cluster:
            if self.attributes[node] == 0 and adj_0 >= 1:
                adj_0 -= 1
            elif self.attributes[node] == 1 and adj_1 >= 1:
                adj_1 -= 1  
            else :
                print(f"Point {cluster} has no effect on cluster {node} due to insufficient count.")
        return adj_0, adj_1  
    
    def calculate_ratio(self, adj_0, adj_1):
        if adj_0 == 0 or adj_1 == 0:
            return 0
        else:
            return min(adj_0 / adj_1, adj_1 / adj_0)
        
    def calculate_majority(self, adj_0, adj_1):
        if adj_0 > adj_1:
            return 0
        elif adj_1 > adj_0:
            return 1
        if adj_0 == adj_1==0:
            return -2
      
    def random_argmax(self, array):
        max_vals = np.max(array, axis=1, keepdims=True)
        is_max = (array == max_vals)
        choices = [np.random.choice(np.flatnonzero(row)) for row in is_max]
        return np.array(choices)
    
    def compute_forces(self, X, centroids):

        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        norms = np.sum(diffs**2, axis=2) 
        norms = np.maximum(norms, 1e-10)
        distances = 1.0 / norms

        if self.lambda_ == 0:
            return distances

        cluster_counts = np.array([
            [self.cluster_attributes[i]['0'], self.cluster_attributes[i]['1']]
            for i in range(self.n_clusters)
        ])  # shape (n_clusters, 2)

        adj_0 = np.broadcast_to(cluster_counts[:,0], (X.shape[0], self.n_clusters))
        adj_1 = np.broadcast_to(cluster_counts[:,1], (X.shape[0], self.n_clusters))

        # imbalance calculation
        ratios = np.where((adj_0 == 0) | (adj_1 == 0), 0,
                        np.minimum(adj_0 / np.maximum(adj_1, 1e-10),
                                    adj_1 / np.maximum(adj_0, 1e-10)))
        imbalance = 1.0 - ratios

        majority_adj = np.where(adj_0 > adj_1, 0, np.where(adj_1 > adj_0, 1, -2))
        direction = np.where(self.attributes[:, None] == majority_adj, -1.0, 1.0)
        direction[majority_adj == -2] = 0

        new_metric = ((1 - self.lambda_) + self.lambda_ * imbalance * direction) * distances
        return new_metric

    def update_centroids(self, X):
        new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if np.sum(self.labels_ == i) > 0 
                else self.centroids[i]
                for i in range(self.n_clusters)
            ])
        return new_centroids

    def fit(self, X, attributes):
        self.attributes = attributes.copy()
        self.centroids = self.initialize_centroids(X)

        l = self.lambda_
        self.lambda_ = 0
        new_distances = self.compute_forces(X, self.centroids)

        # self.labels_ = np.argmax(new_distances, axis=1)
        self.labels_ = self.random_argmax(new_distances)
        self.find_clusters_r_b(self.attributes, self.labels_)
        self.centroids = self.update_centroids(X)  
        self.lambda_ = l 
        iter = 0
        prev_centroids = None

        for _ in range(self.max_iter):
            iter += 1
            new_distances = self.compute_forces(X, self.centroids)
            self.labels_ = self.random_argmax(new_distances)
            self.find_clusters_r_b(self.attributes, self.labels_)
            new_centroids = self.update_centroids(X)
  
            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                self.stop_flag = "convergence"
                print("convergence")
                break

            if prev_centroids is not None and np.allclose(new_centroids, prev_centroids, atol=1e-6):
                self.stop_flag = "oscillation"
                print(f"Oscillation detected at iter {iter}")
                break

            prev_centroids = self.centroids
            self.centroids = new_centroids

        if self.max_iter == iter:
            self.stop_flag = "max_iter"
        self.iter = iter
        
        return self
      
    def compute_new_sse(self, X):
        new_distances = self.compute_forces(X, self.centroids)
        point_errors = new_distances[np.arange(len(X)), self.labels_]
        return np.sum(point_errors)  

    def compute_sse_lambda0(self, X):
        original_lambda = self.lambda_
        try:
            self.lambda_ = 0
            new_distances = self.compute_forces(X, self.centroids)
            point_errors = new_distances[np.arange(len(X)), self.labels_]
            return np.sum(point_errors)
        finally:
            self.lambda_ = original_lambda

    def compute_sse(self, X):
        sse = np.sum([np.linalg.norm(X[i] - self.centroids[self.labels_[i]])**2 for i in range(len(X))])
        return sse

    def compute_silhouette(self, X):

        if len(set(self.labels_)) < 2:
            return 0  
        return silhouette_score(X, self.labels_)
