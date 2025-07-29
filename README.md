# Categorical FeaturesEmbedding 

## 📌 Motivation

One-hot encoding for categorical variables is a widely used technique, especially valued for its ability to eliminate encoding bias. However, representing categories with binary vectors such as `[1 0]`, `[0 1]`, etc., introduces **fixed orthogonal geometry** into the feature space. This rigid structure may not align well with the underlying data distribution and could limit the model's ability to find optimal decision boundaries.

This project proposes a method to learn **low-dimensional, trainable embeddings** for categorical variables by optimizing a geometric criterion—specifically, the **maximum inter-class centroid distance**—using cross-validation. The approach provides a more flexible geometric representation that better reflects the separability of the target classes.

🔬 _Note: This work currently focuses on classification tasks only. Its effectiveness for downstream classifiers is yet to be fully evaluated._
## 📊 Dataset Preparation

The Iris dataset is used as a base, and a categorical feature is artificially created by binning a numeric column:

```python
# Bin petal width into low, medium, high
from sklearn.preprocessing import KBinsDiscretizer
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
X['petal_width_bin'] = binner.fit_transform(X[['petal width (cm)']]).astype(int)
```

This simulates a 3-class categorical column for embedding experiments. 
Objective is still to build a classification model for predicting species but instead of using a numeric col we use the binned feature.

## ⚙️ Core Functions (from `cat_embeddings.py`)

### Objective
- `min_centroid_distance_with_l2(...)`: Maximize minimum centroid distance with L2 penalty

### Embedding Optimization
- `optimize_embedding(...)`: Learns embeddings for given hyperparameters  
- `cross_validated_score(...)`: CV score for a single config  
- `cross_validated_embedding_grid_search(...)`: Full grid search over dimensions, penalties, and bounds

### Embedding Evaluation & Integration
- `create_augmented_dataframe(...)`: Combines scaled numeric + learned embeddings + target  
- `extract_best_config_per_dim(...)`: Extracts best config per embedding dimension  
- `compute_silhouette_scores_per_dim(...)`: Computes silhouette scores from best config  
- `plot_silhouette_scores_per_dim(...)`: Plots silhouette scores per dimension  
- `compare_silhouette_embedding_vs_onehot(...)`: Compares embeddings vs one-hot via silhouette  

## 📈 Results Summary

- Embeddings typically outperform one-hot encoding in silhouette score  
- Visualization shows geometric misalignment with one-hot and better continuity with embeddings  
- Hypothesis testing confirms statistical significance of performance gain in many trials  

## 🗂️ File Structure

```
├── cat_embeddings.py             # All core functions
├── cat_embeddings_results.ipynb # Notebook with experiments
└── README.md                     # Project documentation
```

## ✅ Future Extensions

- Add adaptive logic to decide when to embed  
- Extend to multiple categorical features  
- Integrate with AutoML pipelines  

---

### 🔧 Sample Usage

```python
from cat_embeddings import (
    min_centroid_distance_with_l2,
    cross_validated_embedding_grid_search,
    extract_best_config_per_dim,
    compute_silhouette_scores_per_dim,
    plot_silhouette_scores_per_dim
)

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and prepare data
iris = load_iris()
X = iris.data
y = iris.target
bin_column = np.digitize(X[:, 0], bins=[5.0, 6.0])
scaled_X = StandardScaler().fit_transform(X)

# Grid search over hyperparams
grid_results = cross_validated_embedding_grid_search(
    X_numeric=scaled_X,
    bin_column=bin_column,
    y=y,
    embedding_dims=[1, 2, 3],
    penalties=[0.0, 0.1],
    bounds_abs=[1.0, 2.5],
    objective_fn=min_centroid_distance_with_l2,
    k=5
)

# Best config and silhouette score per dimension
best_per_dim = extract_best_config_per_dim(grid_results)
sil_scores = compute_silhouette_scores_per_dim(
    grid_results=grid_results,
    X_numeric=scaled_X,
    bin_column=bin_column,
    y=y,
    scale_numeric=True
)

plot_silhouette_scores_per_dim(sil_scores)
```
