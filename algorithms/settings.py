import numpy as np
from scipy.sparse import issparse

def l1_norm(x, d):
    # x = (w, b); exclude bias
    return np.sum(np.abs(x[:-1])) / d

def subgradient_l1_norm(x, d):
    # sign subgradient; break ties at 0
    sub_grad = np.sign(x).astype(np.float32)
    sub_grad[-1] = 0.0
    return sub_grad / d

def hinge_loss(X, y, x):
    # average hinge loss
    n = X.shape[0]
    w, b = x[:-1], x[-1]
    preds = X.dot(w) + b  # works for dense or sparse X
    margins = 1 - y * preds
    return float(np.sum(np.maximum(0.0, margins)) / n)

def subgradient_hinge_loss(X, y, x):
    # subgrad for hinge: for i with margin>0, -y_i * a_i; include bias
    n = X.shape[0]
    w, b = x[:-1], x[-1]
    preds = X.dot(w) + b
    margins = 1 - y * preds
    active = (margins > 0).astype(np.float32)
    vec = active * y  # length n
    if issparse(X):
        grad_w = -(X.T @ vec)
        grad_w = np.asarray(grad_w).ravel()
    else:
        grad_w = -(X.T.dot(vec))
    grad_b = -np.dot(active, y)
    grad = np.concatenate([grad_w, np.array([grad_b])]) / n
    return grad

def project_onto_box(x, bound):
    return np.clip(x, -bound, bound)
