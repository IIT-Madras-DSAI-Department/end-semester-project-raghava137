import numpy as np
from scipy.stats import mode
import numpy as np
import pandas as pd
import time
class optiKNN:
    def __init__(self, k=5):
        self.k = k
        self.Xtrain = None
        self.ytrain = None

    def fit(self, X, y):
        self.Xtrain = X.astype(np.float32) 
        self.ytrain = y.astype(int)
        self.train_sq_norms = np.sum(self.Xtrain**2, axis=1).reshape(1, -1)

    def _euclidean_distance(self, x):
        return np.sum((self.Xtrain - x)**2, axis=1)

    def predict_one(self, x):
        distances = self._euclidean_distance(x)
        idx = np.argsort(distances)[:self.k]
        neighbors = self.ytrain[idx]
        values, counts = np.unique(neighbors, return_counts=True)
        return values[np.argmax(counts)] 
    def predict(self, X):
        X_test = X.astype(np.float32)
        test_sq_norms = np.sum(X_test**2, axis=1).reshape(-1, 1)
        dot_product = X_test @ self.Xtrain.T
        dists = test_sq_norms + self.train_sq_norms - 2 * dot_product
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        k_nearest_labels = self.ytrain[k_indices]
        preds, _ = mode(k_nearest_labels, axis=1)
        return np.squeeze(preds)
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.Xtrain = None
        self.ytrain = None

    def fit(self, X, y):
        self.Xtrain = X.astype(np.float32)
        self.ytrain = y.astype(int)

    def _euclidean_distance(self, x):
        return np.sum((self.Xtrain - x)**2, axis=1)

    def predict_one(self, x):
        distances = self._euclidean_distance(x)
        idx = np.argsort(distances)[:self.k]
        neighbors = self.ytrain[idx]

        # Majority vote
        values, counts = np.unique(neighbors, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            preds.append(self.predict_one(X[i]))
        return np.array(preds)
class WeightedKNN:
    def __init__(self, k=5):
        self.k = k
        self.Xtrain = None
        self.ytrain = None
        self.train_sq_norms = None
        self.n_classes = None
        self.epsilon = 1e-6 

    def fit(self, X, y):
        self.Xtrain = X.astype(np.float32) 
        self.ytrain = y.astype(int)
        self.train_sq_norms = np.sum(self.Xtrain**2, axis=1).reshape(1, -1)
        self.n_classes = len(np.unique(self.ytrain))
        if self.n_classes <= np.max(self.ytrain):
            self.n_classes = np.max(self.ytrain) + 1

    def _euclidean_distance(self, x):

        return np.sum((self.Xtrain - x)**2, axis=1)

    def predict_one(self, x):
        sq_dists = self._euclidean_distance(x)
        idx = np.argsort(sq_dists)[:self.k]
        neighbors_labels = self.ytrain[idx]
        neighbors_dists = np.sqrt(sq_dists[idx])
        neighbors_weights = 1.0 / (neighbors_dists + self.epsilon)
        total_weights = np.zeros(self.n_classes, dtype=np.float32)
        np.add.at(total_weights, neighbors_labels, neighbors_weights)
        return np.argmax(total_weights)

    def predict(self, X):
        X_test = X.astype(np.float32)
        
        # --- 1. Calculate Distance Matrix (No change) ---
        test_sq_norms = np.sum(X_test**2, axis=1).reshape(-1, 1)
        dot_product = X_test @ self.Xtrain.T
        
        # Dists is (N_test, M_train) matrix of *squared* distances
        dists = test_sq_norms + self.train_sq_norms - 2 * dot_product
        
        # --- 2. Get K-Nearest Labels AND Distances ---
        
        # Find indices of k-nearest (N_test, k)
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        
        # Get labels of k-nearest (N_test, k)
        k_nearest_labels = self.ytrain[k_indices]
        
        # Get the *squared* distances of k-nearest (N_test, k)
        # We use take_along_axis to select the distances using our k_indices
        k_nearest_sq_dists = np.take_along_axis(dists, k_indices, axis=1)
        
        # --- 3. Calculate Weights (NEW) ---
        
        # Handle floating point errors that might make sq_dists < 0
        k_nearest_sq_dists = np.maximum(k_nearest_sq_dists, 0)
        
        # Get the true distances
        k_nearest_dists = np.sqrt(k_nearest_sq_dists)
        
        # Calculate weights (N_test, k)
        k_nearest_weights = 1.0 / (k_nearest_dists + self.epsilon)
        
        # --- 4. Perform Vectorized Weighted Vote (NEW) ---
        
        n_test = X.shape[0]
        
        # This will be our (N_test, n_classes) "score" matrix
        total_weights = np.zeros((n_test, self.n_classes), dtype=np.float32)
        
        # Create a (N_test, 1) array of row indices [0], [1], ..., [N-1]
        row_idx_broadcast = np.arange(n_test)[:, None]
        
        # This is the magic!
        # np.add.at adds weights to the (row, label) position.
        # For each sample, it sums the weights for each class.
        np.add.at(
            total_weights, 
            (row_idx_broadcast, k_nearest_labels), 
            k_nearest_weights
        )
        
        # --- 5. Get Final Predictions (NEW) ---
        
        # Find the class (column) with the maximum total weight for each row
        preds = np.argmax(total_weights, axis=1)
        
        return preds
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        # Initialize all these as None
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit PCA on the dataset X.
        """
        # 0. Convert to array
        X = np.array(X, dtype=float)
        
        # 1. Center the data
        # compute mean 
        self.mean = np.mean(X, axis=0)
        # center around mean
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigen decomposition
        # compute the eigen values and vectors of covariance martix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvectors by descending eigenvalues
        # sort in descending order of eigen values
        sorted_idx = np.argsort(eigenvalues)[::-1]

        # take top n components
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]

        # get the components 
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):
        # center the data around zero
        X_centered = X - self.mean
        
        return np.dot(X_centered, self.components)

    def reconstruct(self, X):

        Z = self.predict(X)  
        return np.dot(Z, self.components.T) + self.mean
    def detect_anomalies(self, X, threshold=None, return_errors=False):
        X_reconstructed = self.reconstruct(X)


        errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        if threshold is None:
            threshold = np.percentile(errors, 95)
        flag = errors > threshold
        
        is_anomaly = flag * 1
        
        return is_anomaly, errors
class FastSoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=200, batch_size=64):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.n_classes = None

    def _softmax(self, z):

        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):

        return np.eye(num_classes)[y]

    def _calculate_loss(self, X, y):
        # Fast loss calculation without creating a giant one-hot matrix
        num_samples = X.shape[0]
        logits = np.dot(X, self.W) + self.b
        probs = self._softmax(logits)
        
        # Pick the log-probability of the correct class for each sample
        correct_log_probs = -np.log(probs[np.arange(num_samples), y] + 1e-15)
        loss = np.mean(correct_log_probs)
        return loss

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.n_classes = np.max(y) + 1

        # Initialize weights and bias
        self.W = np.random.randn(num_features, self.n_classes) * 0.01
        self.b = np.zeros((1, self.n_classes))
        for epoch in range(self.epochs):

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, self.batch_size):
                # Get the mini-batch
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                batch_n = X_batch.shape[0]

                # One-hot encode only the batch
                Y_onehot_batch = self._one_hot(y_batch, self.n_classes)

                logits = np.dot(X_batch, self.W) + self.b
                probs = self._softmax(logits)

                grad_logits = (1. / batch_n) * (probs - Y_onehot_batch)
                grad_W = np.dot(X_batch.T, grad_logits)
                grad_b = np.sum(grad_logits, axis=0, keepdims=True)
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b
            
            # Log loss at the end of the epoch
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                loss = self._calculate_loss(X, y)
                print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
class DecisionTree:
    __slots__ = (
        "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "n_classes_", "n_features_", "rng",
        "tree_", "X_", "y_"
    )

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features="sqrt", random_state=46):
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.rng=np.random.RandomState(random_state)

        self.n_classes_ = None
        self.n_features_ = None
        self.X_ = None  
        self.y_ = None 

        self.tree_ = {} 

    # ------- helpers -------
    def _majority_proba(self, y_indices, n_classes):
        if y_indices.size == 0:
            return np.ones(n_classes) / n_classes
        counts = np.bincount(self.y_[y_indices], minlength=n_classes).astype(float)
        return counts / counts.sum()

    def _gini_from_counts(self, counts):
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        return 1.0 - np.sum(p * p)

    def _choose_feature_subset(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'")
        elif isinstance(self.max_features, float):
            if not (0.0 < self.max_features <= 1.0):
                raise ValueError("max_features float must be in (0, 1].")
            k = max(1, int(self.max_features * n_features))
        else:
            k = int(self.max_features)
            if k <= 0 or k > n_features:
                k = n_features
        return self.rng.choice(n_features, size=k, replace=False)

    # -------- exact split search (sort + scan all change points) --------
    def _best_split_for_feature(self, X_col_subset, y_subset, n_classes, min_leaf):
        """
        This function is already highly optimized (vectorized Gini calc).
        It takes 1D arrays (copies) which is fine and fast.
        """
        n = y_subset.shape[0]
        if n < 2 * min_leaf:
            return np.inf, None, None

        order = np.argsort(X_col_subset, kind="mergesort")
        x_sorted = X_col_subset[order]
        y_sorted = y_subset[order]

        if x_sorted[0] == x_sorted[-1]:
            return np.inf, None, None

        left_counts = np.zeros((n, n_classes), dtype=np.int32)
        for c in range(n_classes):
            left_counts[:, c] = np.cumsum((y_sorted == c).astype(np.int32))
        total_counts = left_counts[-1].copy()

        diffs = np.diff(x_sorted)
        valid_pos = np.where(diffs > 0)[0]
        if valid_pos.size == 0:
            return np.inf, None, None

        left_sizes = valid_pos + 1
        right_sizes = n - left_sizes
        ok = (left_sizes >= min_leaf) & (right_sizes >= min_leaf)
        valid_pos = valid_pos[ok]
        if valid_pos.size == 0:
            return np.inf, None, None

        lc = left_counts[valid_pos]
        rc = total_counts[None, :] - lc
        lsz = left_sizes[ok][:, None].astype(float)
        rsz = right_sizes[ok][:, None].astype(float)
        n_tot = (lsz + rsz)

        l_gini = 1.0 - np.sum((lc / lsz) ** 2, axis=1)
        r_gini = 1.0 - np.sum((rc / rsz) ** 2, axis=1)
        weighted = (lsz[:, 0] / n_tot[:, 0]) * l_gini + (rsz[:, 0] / n_tot[:, 0]) * r_gini

        best_idx = np.argmin(weighted)
        best_pos = valid_pos[best_idx]
        thr = 0.5 * (x_sorted[best_pos] + x_sorted[best_pos + 1])

        left_mask = X_col_subset <= thr
        if (np.sum(left_mask) < min_leaf) or (n - np.sum(left_mask) < min_leaf):
            return np.inf, None, None

        return weighted[best_idx], float(thr), left_mask

    def _best_split(self, node_X_idx):
        n_samples = node_X_idx.size
        n_classes = self.n_classes_
        y_node = self.y_[node_X_idx]
        feat_idx = self._choose_feature_subset(self.n_features_)
        parent_counts = np.bincount(y_node, minlength=n_classes)
        parent_gini = self._gini_from_counts(parent_counts)
        best = (np.inf, None, None, None)
        if parent_gini == 0.0:
            return best
        for j in feat_idx:
            X_col_node = self.X_[node_X_idx, j]
            impurity, thr, left_mask = self._best_split_for_feature(
                X_col_node, y_node, n_classes, self.min_samples_leaf
            )
            if impurity < best[0]:
                best = (impurity, j, thr, left_mask)

        return best


    def fit(self, X, y, sample_indices=None, n_classes=None):

        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y, dtype=np.int32)
        self.n_features_ = self.X_.shape[1]

        if n_classes is None:
            n_classes = int(np.max(self.y_)) + 1
        self.n_classes_ = n_classes

        if sample_indices is None:
            root_indices = np.arange(self.X_.shape[0])
        else:
            root_indices = sample_indices
        

        nodes_list = []

        def _build(node_X_idx, depth):
            y_node_size = node_X_idx.size

            y_node_for_stats = self.y_[node_X_idx]
            proba = self._majority_proba(node_X_idx, self.n_classes_)

            if (self.max_depth is not None and depth >= self.max_depth) \
                or (y_node_size < self.min_samples_split) \
                or (np.unique(y_node_for_stats).size == 1):
                
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            impurity, feat, thr, left_mask = self._best_split(node_X_idx)

            if feat is None:
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            left_idx_global = node_X_idx[left_mask]
            right_idx_global = node_X_idx[~left_mask]

            if left_idx_global.size < self.min_samples_leaf or right_idx_global.size < self.min_samples_leaf:
                node_id = len(nodes_list)
                nodes_list.append({
                    "is_leaf": True, "proba": proba, "feat": -1,
                    "thr": -1.0, "left": -1, "right": -1
                })
                return node_id

            left_child = _build(left_idx_global, depth + 1)
            right_child = _build(right_idx_global, depth + 1)

            node_id = len(nodes_list)
            nodes_list.append({
                "is_leaf": False,
                "proba": proba,
                "feat": feat,
                "thr": float(thr),
                "left": left_child,
                "right": right_child
            })
            return node_id
        root_id = _build(root_indices, depth=0)
        n_nodes = len(nodes_list)
        self.tree_['is_leaf'] = np.zeros(n_nodes, dtype=bool)
        self.tree_['proba'] = np.zeros((n_nodes, self.n_classes_), dtype=float)
        self.tree_['feat'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['thr'] = np.zeros(n_nodes, dtype=float)
        self.tree_['left'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['right'] = np.full(n_nodes, -1, dtype=np.int32)
        self.tree_['root'] = root_id

        for i, node_d in enumerate(nodes_list):
            self.tree_['is_leaf'][i] = node_d['is_leaf']
            self.tree_['proba'][i, :] = node_d['proba']
            self.tree_['feat'][i] = node_d['feat']
            self.tree_['thr'][i] = node_d['thr']
            self.tree_['left'][i] = node_d['left']
            self.tree_['right'][i] = node_d['right']

        del self.X_
        del self.y_
        return self

    def predict_proba(self, X):

        X = np.asarray(X)
        n_samples = X.shape[0]
        is_leaf = self.tree_['is_leaf']
        feat = self.tree_['feat']
        thr = self.tree_['thr']
        left = self.tree_['left']
        right = self.tree_['right']
        proba = self.tree_['proba']
        root = self.tree_['root']

        node_ids = np.full(n_samples, root, dtype=np.int32)

        probas = np.zeros((n_samples, self.n_classes_))
        active = np.ones(n_samples, dtype=bool)
        while np.any(active):

            active_idx = np.where(active)[0]
            curr_nodes = node_ids[active_idx]

            leaves_mask = is_leaf[curr_nodes]

            if np.any(leaves_mask):
 
                leaf_indices = active_idx[leaves_mask]
                
                probas[leaf_indices] = proba[node_ids[leaf_indices]]
                
                active[leaf_indices] = False

            if not np.any(active):
                break # All done

            # Get the *original* indices of non-leaf samples
            non_leaf_mask = ~leaves_mask
            non_leaf_indices = active_idx[non_leaf_mask]
            
            # Get the node IDs for *only* the non-leaf samples
            non_leaf_node_ids = node_ids[non_leaf_indices]

            # Get features and thresholds for these nodes
            node_feats = feat[non_leaf_node_ids]
            node_thrs = thr[non_leaf_node_ids]
            
            # Get the corresponding data from X and make the decision
            X_data = X[non_leaf_indices, node_feats]
            go_left_mask = X_data <= node_thrs

            # Get the global indices for left/right branches
            left_global_indices = non_leaf_indices[go_left_mask]
            right_global_indices = non_leaf_indices[~go_left_mask]
            
            # Get the children node IDs
            left_child_nodes = left[non_leaf_node_ids[go_left_mask]]
            right_child_nodes = right[non_leaf_node_ids[~go_left_mask]]

            # Update node_ids for the next iteration
            node_ids[left_global_indices] = left_child_nodes
            node_ids[right_global_indices] = right_child_nodes
            
        return probas
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
class RandomForest:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 max_features="sqrt",
                 min_samples_leaf=1,
                 min_samples_split=2,
                 bootstrap=True,
                 max_samples=None,
                 random_state=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.random_state = random_state

        self.n_classes_ = None
        self.trees_ = []
        self.rng = np.random.RandomState(random_state) 
    def _draw_sample_indices(self, n_samples):
        if self.max_samples is None:
            m = n_samples
        elif isinstance(self.max_samples, float):
            m = max(1, int(self.max_samples * n_samples))
        else:
            m = int(self.max_samples)
            if m <= 0 or m > n_samples:
                m = n_samples
        if self.bootstrap:
            return self.rng.randint(0, n_samples, size=m) 
        else:
            return self.rng.choice(n_samples, size=m, replace=False)
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.int32)
        self.n_classes_ = int(np.max(y)) + 1
        self.trees_ = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            print(f"Fitting tree {_ + 1}/{self.n_estimators}")
            idx = self._draw_sample_indices(n_samples)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.rng.randint(0, 2**31 - 1)
            )
            tree.fit(X, y, sample_indices=idx, n_classes=self.n_classes_)
            self.trees_.append(tree)
        return self
    def predict_proba(self, X):
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree in self.trees_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.trees_)
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
class treeNode:
    def __init__(self, threshold=None, feature_index=None, value=None):
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.left = None
        self.right = None

    def is_leaf_Node(self):
        return self.value is not None
import numpy as np

class treeNode:
    def __init__(self, threshold=None, feature_index=None, value=None):
        self.threshold = threshold
        self.feature_index = feature_index
        self.value = value
        self.left = None
        self.right = None

    def is_leaf_Node(self):
        return self.value is not None

class BinaryXGB:
    def __init__(self, n_estimators=120, learning_rate=0.5, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.trees = []
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def _g(self, y_true, y_pred):
        return self._sigmoid(y_pred) - y_true
    def _h(self, y_true, y_pred):
        sig = self._sigmoid(y_pred)
        return sig * (1 - sig)
    def _exact_greedysplit_vectorized(self, X_col, y_true, y_pred):
        g = self._g(y_true, y_pred)
        h = self._h(y_true, y_pred)
        nonzero_mask = X_col != 0
        X_nz = X_col[nonzero_mask]
        g_nz = g[nonzero_mask]
        h_nz = h[nonzero_mask]

        if X_nz.size < 2:
            return -np.inf, None
        G_total, H_total = np.sum(g), np.sum(h)
        G_zero = np.sum(g[~nonzero_mask])
        H_zero = np.sum(h[~nonzero_mask])
        sorted_idx = np.argsort(X_nz)
        X_sorted = X_nz[sorted_idx]
        g_sorted = g_nz[sorted_idx]
        h_sorted = h_nz[sorted_idx]
        G_L = G_zero + np.cumsum(g_sorted)
        H_L = H_zero + np.cumsum(h_sorted)
        G_R = G_total - G_L
        H_R = H_total - H_L
        gain = (G_L**2) / (H_L + self.lamda + 1e-6) + \
               (G_R**2) / (H_R + self.lamda + 1e-6) - \
               (G_total**2) / (H_total + self.lamda + 1e-6)
        best_idx = np.argmax(gain)
        best_gain = gain[best_idx]
        best_threshold = X_sorted[best_idx]
        return best_gain, best_threshold
    def _build_tree(self, X, y_true, y_pred, depth):
        n_samples, n_features = X.shape
        if (n_samples < 3) or (depth >= self.max_depth):
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)
        feature_indices = np.random.choice(
            n_features,
            int(max(1, self.subsample_features * n_features)),
            replace=False
        )
        best_gain, best_threshold, best_feature = -np.inf, None, None
        for feature_index in feature_indices:
            gain, threshold = self._exact_greedysplit_vectorized(X[:, feature_index], y_true, y_pred)
            if gain > best_gain:
                best_gain, best_threshold, best_feature = gain, threshold, feature_index
        if best_gain < 1e-6:
            G = np.sum(self._g(y_true, y_pred))
            H = np.sum(self._h(y_true, y_pred))
            leaf_value = -G / (H + self.lamda + 1e-6)
            return treeNode(value=leaf_value)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_mask], y_true[left_mask], y_pred[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y_true[right_mask], y_pred[right_mask], depth + 1)
        node = treeNode(threshold=best_threshold, feature_index=best_feature)
        node.left = left_subtree
        node.right = right_subtree
        return node
    def _predict_tree(self, X, tree):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = tree
            while not node.is_leaf_Node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.value
        return y_pred
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        y_mean = np.mean(y)
        y_pred = np.full(y.shape, np.log(y_mean / (1 - y_mean + 1e-6)))
        for _ in range(self.n_estimators):
            tree = self._build_tree(X, y, y_pred, 0)
            self.trees.append(tree)
            update = self._predict_tree(X, tree)
            y_pred += self.learning_rate * update
    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        y_pred = self._sigmoid(y_pred)
        return y_pred
class MultiClassXGB:
    def __init__(self, n_estimators=50, learning_rate=0.3, max_depth=6,
                 lamda=3.0, subsample_features=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lamda = lamda
        self.subsample_features = subsample_features
        self.classifiers = []
        self.classes_ = None
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.classifiers = []
        for i, cls in enumerate(self.classes_):
            y_binary = (y == cls).astype(int)
            classifier = BinaryXGB(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                lamda=self.lamda,
                subsample_features=self.subsample_features
            )
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classifiers)
        proba_matrix = np.zeros((n_samples, n_classes))
        for i, classifier in enumerate(self.classifiers):
            proba_matrix[:, i] = classifier.predict(X)
        proba_sum = np.sum(proba_matrix, axis=1, keepdims=True)
        proba_matrix = proba_matrix / (proba_sum + 1e-10)
        return proba_matrix
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
  
class KNNEnsemble:
    def __init__(self, val_ratio=0.3):
        self.k_values_first = [5, 7, 3, 4, 6]
        self.k_values_second = [6, 8, 5]
        self.n_components = [49, 64, 35]
        self.val_ratio = val_ratio
        self.model_weights = {}
    def fit(self, Xtrain, ytrain):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        n_val = int(len(Xtrain) * self.val_ratio)
        indices = np.random.permutation(len(Xtrain))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_val, y_val = Xtrain[val_idx], ytrain[val_idx]
        X_tr, y_tr = Xtrain[train_idx], ytrain[train_idx]
        for k in self.k_values_first:
            model = optiKNN(k=k)
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_val)
            accuracy = np.mean(val_preds == y_val)
            self.model_weights[f'knn_{k}'] = accuracy ** 2 
        for k in self.k_values_second:
            for n in self.n_components:
                pca = PCAModel(n_components=n)
                pca.fit(X_tr)
                X_tr_pca = pca.predict(X_tr)
                X_val_pca = pca.predict(X_val)
                model = WeightedKNN(k=k)
                model.fit(X_tr_pca, y_tr)
                val_preds = model.predict(X_val_pca)
                accuracy = np.mean(val_preds == y_val)
                self.model_weights[f'pca_{k}_{n}'] = accuracy ** 2
    def predict(self, Xtest):
        all_predictions = []
        all_weights = []
        for k in self.k_values_first:
            model = optiKNN(k=k)
            model.fit(self.Xtrain, self.ytrain)
            preds = model.predict(Xtest)
            weight = self.model_weights[f'knn_{k}']  
            all_predictions.append(preds)
            all_weights.extend([weight] * len(preds))
        for k in self.k_values_second:
            for n in self.n_components:
                pca = PCAModel(n_components=n)
                pca.fit(self.Xtrain)
                Xtest_pca = pca.predict(Xtest)
                model = WeightedKNN(k=k)
                model.fit(pca.predict(self.Xtrain), self.ytrain)
                preds = model.predict(Xtest_pca)
                weight = self.model_weights[f'pca_{k}_{n}']
                copies = 9 if (k == 6 and n == 49) else 1
                for _ in range(copies):
                    all_predictions.append(preds)
                    all_weights.extend([weight] * len(preds))
        return self._weighted_mode(all_predictions, all_weights)
    def _weighted_mode(self, predictions, weights):
        n_samples = len(predictions[0])
        final_predictions = []
        for i in range(n_samples):
            votes = {}
            for j, preds in enumerate(predictions):
                pred = preds[i]
                weight = weights[j * n_samples + i]  
                votes[pred] = votes.get(pred, 0) + weight
            final_predictions.append(max(votes.items(), key=lambda x: x[1])[0])
        return np.array(final_predictions)
class StackedEnsemble:
    def __init__(self, val_ratio=0.3):
        self.val_ratio = val_ratio
        self.base_models_to_fit = [
            ("SmartEnsemble", KNNEnsemble(val_ratio=0.3)),
            ("RF", RandomForest(
                 n_estimators=120,
                 max_depth=15,
                 max_features="sqrt",
                 min_samples_leaf=1,
                 min_samples_split=2,
                 bootstrap=True,
                 max_samples=None,
                 random_state=42)),
            ("XGB", MultiClassXGB(n_estimators=50, learning_rate=0.3, max_depth=6,
                 lamda=3.0, subsample_features=0.1)) 
        ]
        self.base_models_final = []
        self.meta_model = FastSoftmaxRegression(learning_rate=0.05, epochs=100, batch_size=32)
    def _one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]
    def fit(self, X, y):
        n_classes = np.max(y) + 1
        n_val = int(len(X) * self.val_ratio)
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        meta_features = []
        for name, model in self.base_models_to_fit:
            if model is None: continue

            print(f"  Training {name}...")
            start_fit = time.time()
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_val)
            meta_features.append(val_preds)
            print(f"    {name} fit complete in {time.time() - start_fit:.2f}s")
        meta_features_ohe = [self._one_hot(p, n_classes) for p in meta_features]
        X_meta = np.hstack(meta_features_ohe)
        y_meta = y_val
        
        # Normalize (good for Softmax)
        X_meta_norm = X_meta / X_meta.max()
        
        self.meta_model.fit(X_meta_norm, y_meta)

        self.base_models_final = []
        for name, model in self.base_models_to_fit:
            if model is None: continue
 
            start_fit = time.time()
            model.fit(X, y)
            self.base_models_final.append(model)
    def predict(self, X):
        n_classes = self.meta_model.n_classes
        meta_features = []
        for model in self.base_models_final:
            preds = model.predict(X)
            meta_features.append(preds)
        meta_features_ohe = [self._one_hot(p, n_classes) for p in meta_features]
        X_meta_test = np.hstack(meta_features_ohe)
        X_meta_test_norm = X_meta_test / X_meta_test.max()
        return self.meta_model.predict(X_meta_test_norm)
