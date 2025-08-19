import os
import pickle

def save(path="results", *args, **kwargs):
    print(kwargs)
    if not os.path.exists(path):
        os.makedirs(path)

    for i, arg in enumerate(args):
        full_path = os.path.join(path, f"{kwargs[i]}.pkl")
        with open(full_path, 'wb') as f:
            pickle.dump(arg, f)
        print(f"Saved {kwargs[i]} to {full_path}")