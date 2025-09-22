import os
import argparse
import json
import numpy as np
from utils import load, plot


def _to_numeric(arr_like):
    # Drop None entries and cast to float ndarray
    if arr_like is None:
        return np.array([], dtype=float)
    seq = [v for v in list(arr_like) if v is not None]
    return np.asarray(seq, dtype=float)


def extract_series(payload):
    if isinstance(payload, dict):
        f = _to_numeric(payload.get('f', []))
        g = _to_numeric(payload.get('g', []))
        return f, g

    # Backward compatibility: handle old pickled solver objects
    if hasattr(payload, 'f_plot') and hasattr(payload, 'g_plot'):
        f = _to_numeric(getattr(payload, 'f_plot'))
        g = _to_numeric(getattr(payload, 'g_plot'))
    else:
        f = _to_numeric(getattr(payload, 'l1_norm_history', []))
        g = _to_numeric(getattr(payload, 'hinge_loss_history', []))
    return f, g


def resolve_base_name(name: str) -> str:
    # Map display names to file base names
    mapping = {
        'FC-BiO': 'FCBiO',
        'a-IRG': 'aIRG',
    }
    return mapping.get(name, name)


def candidate_files(base: str):
    return [f"{base}.json", f"{base}.pkl"]


def main():
    ap = argparse.ArgumentParser(description='Compare and plot results from saved runs')
    ap.add_argument('--results-dir', type=str, required=True, help='Directory containing saved metrics (JSON) and baselines.json')
    ap.add_argument('--algos', type=str, nargs='+', default=['Bi-CS-RL','Bi-CS-R','Bi-CS-N','Bi-CS-ER','FC-BiO','a-IRG','IIBA'], help='Algorithm names to compare')
    args = ap.parse_args()

    res_dir = args.results_dir
    base_path = os.path.join(res_dir, 'baselines.json')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Baseline file not found: {base_path}')
    with open(base_path, 'r') as f:
        base = json.load(f)
    g_opt = float(base['g_opt'])
    f_opt = float(base['f_opt'])

    series = {}
    for name in args.algos:
        base = resolve_base_name(name)
        obj = None
        chosen = None
        for candidate in candidate_files(base):
            full = os.path.join(res_dir, candidate)
            if os.path.exists(full):
                chosen = candidate
                obj = load(candidate, path=res_dir)
                break
        if obj is None:
            print(f'Warning: missing results for {name}; expected one of {candidate_files(base)} in {res_dir}. Skipping.')
            continue

        f, g = extract_series(obj)
        series[name] = (f, g)

    if len(series) == 0:
        print('No series loaded. Exiting.')
        return

    # Align lengths to min length for plotting
    min_len = min(len(v[0]) for v in series.values())
    names = list(series.keys())
    f_gaps = [series[n][0][:min_len] - f_opt for n in names]
    g_gaps = [series[n][1][:min_len] - g_opt for n in names]
    label_map = {str(i): names[i] for i in range(len(names))}

    upper_path = os.path.join(res_dir, 'plot_upper.png')
    lower_path = os.path.join(res_dir, 'plot_lower.png')

    plot(upper_path, 'Upper Gap', *f_gaps, **label_map)
    print(f'Saved {upper_path}')
    plot(lower_path, 'Lower Gap', *g_gaps, **label_map)
    print(f'Saved {lower_path}')


if __name__ == '__main__':
    main()
