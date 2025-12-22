# Knowledge Graph Implementation Details

This document provides technical details on the Knowledge Graph (KG) generation logic implemented in `Gallblader_Cancer.ipynb`. The notebook automatically generates two types of graphs to help interpret model behavior and training dynamics.

## 1. Model-Concept Graph

The **Model-Concept Graph** is designed to visualize how the Deep Learning models group inputs in their internal feature space, effectively surfacing "learned concepts".

### Purpose
- **Interpretability**: Reveals if the model is learning meaningful clusters (e.g., separating malignant vs. benign features).
- **Debugging**: Identifies "confused" clusters where the model mixes different classes.
- **Visual Proof**: Provides a graph representation of the model's internal state.

### Construction Logic

The graph generation process (implemented in `build_and_save_model_concept_graph`) follows these steps:

1.  **Feature Extraction**:
    - The code iterates through the test set.
    - Extract features from the model's feature extractor (before the final classification layer).
    - Features are flattened into 1D vectors for every image.

2.  **Dimensionality Reduction**:
    - **StandardScaler**: Normalizes the feature vectors.
    - **PCA**: Reduces dimensionality to a maximum of 50 components to make clustering robust.

3.  **Clustering**:
    The system groups similar images into "concepts" (clusters). The clustering logic supports three modes (`n_clusters` parameter):
    - **`'auto'`** (Default): Automatically selects the best number of clusters ($k$) between 2 and 12 using the **Silhouette Score**.
    - **`'all'`**: Treats every single sample as its own unique cluster (useful for small datasets or detailed instance-based views).
    - **`int`**: Uses a fixed number of clusters (e.g., `k=6`) with KMeans.

### Graph Schema (NetworkX)

The resulting graph is a **bipartite-like** structure connecting models to learned clusters, and clusters to specific instances and classes.

| Node Type | ID Format | Description | Attributes |
| :--- | :--- | :--- | :--- |
| **Model** | `model:{name}` | Root node representing the DL model. | `label`: Model Name |
| **Cluster** | `model:cluster:{c}` | Represents a group of similar images (a "concept"). | `support`: # of images<br>`majority_label`: Most common class<br>`majority_support`: Count of majority class |
| **Class** | `class:{label}` | Represents a ground-truth class (e.g., "Benign"). | `label`: Class Name |
| **Image** | `model:img:{i}` | A specific image sample from the dataset. | `path`: File path<br>`true_label`: Ground truth label |

**Edges**:
- `Model` $\rightarrow$ `Cluster`: "This model learned this concept."
- `Cluster` $\rightarrow$ `Class`: "This concept mostly corresponds to this class" (Edge weight = purity/confidence).
- `Cluster` $\rightarrow$ `Image`: "This image belongs to this concept."

### Generated Artifacts
For each model (e.g., `efficientnet_b0`), the following artifacts are saved to `kg_artifacts/`:
- **`{prefix}.graphml`**: The full graph structure (viewable in Gephi, Cytoscape).
- **`{prefix}.png`**: A 2D network visualization of the graph nodes and edges.
- **`{prefix}_pca.png`**: A PCA scatter plot where points are colored by their assigned cluster.
- **`{prefix}_samples/`**: A directory containing sample image thumbnails for each cluster (visual proof of what the cluster represents).

---

## 2. Diagnostic Graph

The **Diagnostic Graph** creates a structural representation of the model's training history.

### Purpose
- **Training Analysis**: Visualizes the progression of loss and accuracy over epochs.
- **Convergence**: Helps identify where the model converged or started overfitting.

### Graph Schema

- **Nodes**: `epoch:{i}` representing state at epoch $i$.
    - **Attributes**: `train_loss`, `train_acc`, `val_loss`, `val_acc`.
- **Edges**: Sequential connections (Epoch $i$ $\rightarrow$ Epoch $i+1$).

### Generated Artifacts
- **`{prefix}.graphml`**: Graph representation of the training history.
- **`{prefix}.png`**: Side-by-side plots of Loss vs. Epoch and Accuracy vs. Epoch.

---

## Usage

These graphs are generated automatically at the end of the training pipeline in `Gallblader_Cancer.ipynb`.

To manually trigger generation for a specific model:

```python
# define output prefix
prefix = f"kg_{model_name}_manual"

# 1. Build Model-Concept Graph
build_and_save_model_concept_graph(
    name=model_name,
    model=model_object,
    dataloader=test_loader,
    device=device,
    out_prefix=prefix,
    n_clusters='auto'
)

# 2. Build Diagnostic Graph (requires history lists)
build_and_save_diagnostic_graph(
    name=model_name,
    train_loss=history['train_loss'],
    train_acc=history['train_acc'],
    val_loss=history['val_loss'],
    val_acc=history['val_acc'],
    out_prefix=prefix
)
```
