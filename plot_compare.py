import os
import argparse
import json
import numpy as np
from utils import load, plot


def extract_series(obj, algo_name):
    if algo_name.lower() == 'bics' or (hasattr(obj, 'f_plot') and hasattr(obj, 'g_plot')):
        f = np.array(getattr(obj, 'f_plot'))
        g = np.array(getattr(obj, 'g_plot'))
    else:
        # assume FCBiO layout
        f = np.array(getattr(obj, 'l1_norm_history'))
        g = np.array(getattr(obj, 'hinge_loss_history'))
    return f, g


def main():
    ap = argparse.ArgumentParser(description='Compare and plot results from saved runs')
    ap.add_argument('--results-dir', type=str, required=True, help='Directory containing saved pickles and baselines.json')
    ap.add_argument('--algos', type=str, nargs='+', default=['BiCS','FCBiO'], help='Default algorithm set')
    ap.add_argument('--upper-algos', type=str, nargs='+', default=None, help='Algorithm names for plot_upper (defaults to --algos)')
    ap.add_argument('--lower-algos', type=str, nargs='+', default=None, help='Algorithm names for plot_lower (defaults to --algos or set difference when --lower-exclude-upper)')
    ap.add_argument('--lower-exclude-upper', action='store_true', help='Exclude upper set from lower set')
    args = ap.parse_args()

    res_dir = args.results_dir
    base_path = os.path.join(res_dir, 'baselines.json')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Baseline file not found: {base_path}')
    with open(base_path, 'r') as f:
        base = json.load(f)
    g_opt = float(base['g_opt'])
    f_opt = float(base['f_opt'])

    # Decide which algorithms to use per plot
    upper_names = args.upper_algos if args.upper_algos is not None else list(args.algos)
    if args.lower_algos is not None:
        lower_names = list(args.lower_algos)
    elif args.lower_exclude_upper:
        lower_names = [n for n in args.algos if n not in set(upper_names)]
    else:
        lower_names = list(upper_names)

    # Load all required series once
    needed = list(dict.fromkeys(upper_names + lower_names))
    series = {}
    for name in needed:
        pkl = f'{name}.pkl'
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

    # Plot upper
    up_loaded = [n for n in upper_names if n in series]
    if up_loaded:
        min_len_up = min(len(series[n][0]) for n in up_loaded)
        f_gaps = [series[n][0][:min_len_up] - f_opt for n in up_loaded]
        label_map_up = {str(i): up_loaded[i] for i in range(len(up_loaded))}
        upper_path = os.path.join(res_dir, 'plot_upper.png')
        plot(upper_path, 'Upper Gap', *f_gaps, **label_map_up)
        print(f'Saved {upper_path}')
    else:
        print('No series available for upper plot.')

    # Plot lower, excluding upper ones if requested
    low_loaded = [n for n in lower_names if n in series]
    if low_loaded:
        min_len_low = min(len(series[n][1]) for n in low_loaded)
        g_gaps = [series[n][1][:min_len_low] - g_opt for n in low_loaded]
        label_map_low = {str(i): low_loaded[i] for i in range(len(low_loaded))}
        lower_path = os.path.join(res_dir, 'plot_lower.png')
        plot(lower_path, 'Lower Gap', *g_gaps, **label_map_low)
        print(f'Saved {lower_path}')
    else:
        print('No series available for lower plot.')


if __name__ == '__main__':
    main()
