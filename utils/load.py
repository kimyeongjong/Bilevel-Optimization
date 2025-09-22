import json
import os
import pickle


def _load_pickle(full_path):
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded pickle data from {full_path}")
    return data


def _load_json(full_path):
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded JSON data from {full_path}")
    return data


def load(filename, path="results"):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File {full_path} does not exist.")

    ext = os.path.splitext(full_path)[1].lower()
    if ext == ".json":
        return _load_json(full_path)
    if ext == ".pkl":
        return _load_pickle(full_path)
    # Fallback: try JSON, then pickle
    try:
        return _load_json(full_path)
    except json.JSONDecodeError:
        return _load_pickle(full_path)
