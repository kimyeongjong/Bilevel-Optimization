import json
import os
import numpy as np


def _to_float_list(seq):
    if seq is None:
        return []
    # Filter out None entries before casting
    filtered = [val for val in seq if val is not None]
    if not filtered:
        return []
    return np.asarray(filtered, dtype=float).tolist()


def _extract_series(obj):
    if hasattr(obj, "f_plot") and hasattr(obj, "g_plot"):
        f_series = getattr(obj, "f_plot", [])
        g_series = getattr(obj, "g_plot", [])
    else:
        f_series = getattr(obj, "l1_norm_history", [])
        g_series = getattr(obj, "hinge_loss_history", [])
    return _to_float_list(f_series), _to_float_list(g_series)


def save(path="results", *args, **kwargs):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, arg in enumerate(args):
        key = kwargs[str(i)]
        f_vals, g_vals = _extract_series(arg)
        payload = {
            "name": key,
            "f": f_vals,
            "g": g_vals,
        }
        full_path = os.path.join(path, f"{key}.json")
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"Saved {key} metrics to {full_path}")
