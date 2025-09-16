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


def extract_series(obj, algo_name):
    # Bi-CS/a-IRG/IIBA expose f_plot/g_plot; FC-BiO exposes *_history
    if hasattr(obj, 'f_plot') and hasattr(obj, 'g_plot'):
        f = _to_numeric(getattr(obj, 'f_plot'))
        g = _to_numeric(getattr(obj, 'g_plot'))
    else:
        f = _to_numeric(getattr(obj, 'l1_norm_history', []))
        g = _to_numeric(getattr(obj, 'hinge_loss_history', []))
    return f, g


def resolve_pkl_name(name: str) -> str:
    # Map display names to pickle base filenames
    mapping = {
        'FC-BiO': 'FCBiO',
        'a-IRG': 'aIRG',
    }
    return mapping.get(name, name)


def main():
    ap = argparse.ArgumentParser(description='Compare and plot results from saved runs')
    ap.add_argument('--results-dir', type=str, required=True, help='Directory containing saved pickles and baselines.json')
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
        pkl = f'{resolve_pkl_name(name)}.pkl'
        full = os.path.join(res_dir, pkl)
        if not os.path.exists(full):
            print(f'Warning: missing {full}; skipping')
            continue
        obj = load(os.path.basename(full), path=res_dir)
        f, g = extract_series(obj, name)
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
