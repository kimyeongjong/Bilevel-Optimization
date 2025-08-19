import os
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix

def load_rcv1_data(n_samples=10000, label_idx=0, path=None):
    """
    Load the RCV1 dataset and return it as a tuple of (X, y) of a sparse matrix and a dense vector.
    The dataset is shuffled and split into features and labels.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, path) if path else here
    full_path = f"{path}/rcv1_{n_samples}samples_idx{label_idx}.npz"
    if path and os.path.exists(full_path):
        data = np.load(full_path, allow_pickle=True)
        X, y = data['X'].item(), data['y']
        print(f"Loaded data from {full_path}")
        return X, y
    else:
        print(f"Loading RCV1 dataset with {n_samples} samples and label index {label_idx}...")
        rcv1 = fetch_rcv1()
        X_all = rcv1.data      # shape: (804414, 47236), sparse matrix
        y_all = rcv1.target  # shape: (804414, 103), multi-label sparse

        y = y_all[:, label_idx].toarray().flatten().astype(np.float32)
        y = 2 * y - 1  # Convert from {0,1} to {-1,+1}


        X, y = shuffle(X_all, y, random_state=42)
        X, y = X[:n_samples], y[:n_samples]

        save_path = os.path.join(path, f"rcv1_{n_samples}samples_idx{label_idx}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(f"{path}/rcv1_{n_samples}samples_idx{label_idx}.npz", X=X, y=y)
        print(f"Saved data to {full_path}")
        print(f"Data shape: X={X.shape}, y={y.shape}")
        return X, y