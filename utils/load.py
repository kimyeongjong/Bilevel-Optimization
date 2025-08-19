import os
import pickle

def load(filename, path="results"):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File {full_path} does not exist.")
    
    with open(full_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from {full_path}")
    return data