import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix

def load_rcv1_data(n_samples=1000, label_idx=0):
    """
    Load the RCV1 dataset and return it as a sparse matrix.
    The dataset is shuffled and split into features and labels.
    """
    rcv1 = fetch_rcv1()
    X_all = rcv1.data      # shape: (804414, 47236), sparse matrix
    y_all = rcv1.target  # shape: (804414, 103), multi-label sparse

    y = y_all[:, label_idx].toarray().flatten().astype(np.float32)
    y = 2 * y - 1  # Convert from {0,1} to {-1,+1}


    X, y = shuffle(X_all, y, random_state=42)
    X, y = X[:n_samples], y[:n_samples]

    return X, y