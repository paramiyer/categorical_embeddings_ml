import numpy as np
from typing import List, Dict, Tuple, Callable
from sklearn.model_selection import KFold
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Objective Function(s)
# ---------------------------------------------------

def min_centroid_distance_with_l2(
    embedding_flat: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_bin: np.ndarray,
    dim: int,
    alpha: float
) -> float:
    """
    Objective: Maximize minimum inter-class centroid distance + L2 regularization.
    """
    embeddings = embedding_flat.reshape(3, dim)
    combined = np.hstack([train_X, embeddings[train_bin]])
    
    centroids = np.array([
        combined[train_y == cls].mean(axis=0)
        for cls in np.unique(train_y)
    ])
    
    dists = pairwise_distances(centroids)
    min_dist = np.min([
        dists[i, j]
        for i in range(len(centroids)) for j in range(i+1, len(centroids))
    ])
    
    penalty = alpha * np.sum(embeddings ** 2)
    return -min_dist + penalty

# ---------------------------------------------------
# Embedding Optimizer
# ---------------------------------------------------

def optimize_embedding(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_bin: np.ndarray,
    dim: int,
    alpha: float,
    bound_abs: float,
    objective_fn: Callable
):
    """
    Minimize objective function to learn optimal embeddings for current fold.
    """
    init_embedding = np.random.randn(3 * dim)
    bounds = [(-bound_abs, bound_abs)] * (3 * dim)

    # Closure-free objective call
    wrapped_objective = lambda emb: objective_fn(
        emb, train_X, train_y, train_bin, dim, alpha
    )

    result = minimize(wrapped_objective, init_embedding, bounds=bounds, method='L-BFGS-B')
    return result

# ---------------------------------------------------
# K-Fold Cross-Validation
# ---------------------------------------------------

def cross_validated_score(
    X_numeric: np.ndarray,
    bin_column: np.ndarray,
    y: np.ndarray,
    dim: int,
    alpha: float,
    bound_abs: float,
    k: int,
    random_state: int,
    objective_fn: Callable
) -> Dict:
    """
    Perform K-fold CV for a given hyperparameter configuration.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_scores = []
    best_score = -np.inf
    best_embedding = None

    for train_idx, _ in kf.split(X_numeric):
        train_X = X_numeric[train_idx]
        train_y = y[train_idx]
        train_bin = bin_column[train_idx]

        result = optimize_embedding(train_X, train_y, train_bin, dim, alpha, bound_abs, objective_fn)
        score = -result.fun
        fold_scores.append(score)

        if score > best_score:
            best_score = score
            best_embedding = result.x.reshape(3, dim)

    return {
        'mean_score': np.mean(fold_scores),
        'std_score': np.std(fold_scores),
        'best_score': best_score,
        'best_embedding': best_embedding
    }

# ---------------------------------------------------
# Grid Search Across Hyperparameters
# ---------------------------------------------------

def cross_validated_embedding_grid_search(
    X_numeric: np.ndarray,
    bin_column: np.ndarray,
    y: np.ndarray,
    embedding_dims: List[int],
    penalties: List[float],
    bounds_abs: List[float],
    objective_fn: Callable,
    k: int = 5,
    random_state: int = 42
) -> Dict[Tuple[int, float, float], Dict]:
    """
    Grid search over embedding dimension, regularization penalty, and bounds.
    """
    results = {}
    for dim in embedding_dims:
        for alpha in penalties:
            for bound_abs in bounds_abs:
                key = (dim, alpha, bound_abs)
                result = cross_validated_score(
                    X_numeric, bin_column, y,
                    dim, alpha, bound_abs,
                    k, random_state, objective_fn
                )
                results[key] = result
    return results

def create_augmented_dataframe(
    X_numeric: np.ndarray,
    bin_column: np.ndarray,
    embedding: np.ndarray,
    y: np.ndarray,
    numeric_feature_names: List[str] = None,
    embedding_prefix: str = "emb",
    scale_numeric: bool = False
) -> pd.DataFrame:
    """
    Constructs a DataFrame with optional scaled numeric features, optimized embedding vectors,
    and the target variable.

    Parameters:
        X_numeric (np.ndarray): Numeric feature array (n_samples, n_features)
        bin_column (np.ndarray): Categorical feature used for embeddings (values 0, 1, 2...)
        embedding (np.ndarray): Learned embedding matrix of shape (n_bins, dim)
        y (np.ndarray): Target array
        numeric_feature_names (List[str], optional): Names for numeric columns
        embedding_prefix (str): Prefix for embedding column names
        scale_numeric (bool): If True, scales numeric features using StandardScaler

    Returns:
        pd.DataFrame: Augmented DataFrame
    """
    if numeric_feature_names is None:
        numeric_feature_names = [f"num_{i}" for i in range(X_numeric.shape[1])]

    X_processed = X_numeric
    if scale_numeric:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_numeric)

    df_numeric = pd.DataFrame(X_processed, columns=numeric_feature_names)

    emb_dim = embedding.shape[1]
    emb_features = embedding[bin_column]  # shape: (n_samples, emb_dim)
    emb_feature_names = [f"{embedding_prefix}_{i}" for i in range(emb_dim)]
    df_emb = pd.DataFrame(emb_features, columns=emb_feature_names)

    df = pd.concat([df_numeric, df_emb], axis=1)
    df["target"] = y

    return df

def extract_best_config_and_embedding_per_dim(
    grid_results: Dict[Tuple[int, float, float], Dict]
) -> Dict[int, Dict]:
    """
    Extract best config and precomputed embedding per dimension from grid_results.

    Returns:
        Dict mapping embedding dim → {
            'config': (dim, alpha, bound_abs),
            'embedding': np.ndarray,
            'score': best_score
        }
    """
    best_per_dim = {}

    for (dim, alpha, bound_abs), result in grid_results.items():
        score = result["best_score"]
        if dim not in best_per_dim or score > best_per_dim[dim]["score"]:
            best_per_dim[dim] = {
                "config": (dim, alpha, bound_abs),
                "embedding": result["best_embedding"],
                "score": score
            }

    return best_per_dim

from sklearn.metrics import silhouette_score

def compute_silhouette_per_dim_from_embeddings(
    best_per_dim: Dict[int, Dict],
    X_numeric: np.ndarray,
    bin_column: np.ndarray,
    y: np.ndarray,
    scale_numeric: bool = True
) -> Dict[int, float]:
    """
    Use precomputed embeddings per dim to augment features and compute silhouette scores.

    Returns:
        Dict[dim] → silhouette score
    """
    silhouette_scores = {}

    for dim, record in best_per_dim.items():
        embedding = record["embedding"]

        df_aug = create_augmented_dataframe(
            X_numeric=X_numeric,
            bin_column=bin_column,
            embedding=embedding,
            y=y,
            scale_numeric=scale_numeric
        )

        X_aug = df_aug.drop(columns="target").values
        sil_score = silhouette_score(X_aug, y)

        silhouette_scores[dim] = sil_score

    return silhouette_scores


def plot_silhouette_scores(sil_scores: Dict[int, float]):
    dims = sorted(sil_scores.keys())
    scores = [sil_scores[dim] for dim in dims]

    plt.figure(figsize=(8, 5))
    plt.plot(dims, scores, marker='o', linestyle='-')
    plt.title("Silhouette Score vs Embedding Dimension")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Silhouette Score (Augmented Features)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
