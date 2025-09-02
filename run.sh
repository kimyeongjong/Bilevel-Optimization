#!/usr/bin/env bash
# Unified runner for BiCS/FCBiO. Computes baselines once, then runs algos
# sequentially or in parallel, and generates comparison plots.
set -euo pipefail

# Defaults (can override via env vars)
NSAMPLES=${NSAMPLES:-10000}
LABEL_IDX=${LABEL_IDX:-0}
BOUND=${BOUND:-50}
DOMAIN=${DOMAIN:-box}
BICS_ITERS=${BICS_ITERS:-1000}
FCBIO_T=${FCBIO_T:-1000}
EPS=${EPS:-0.1}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results/results_${NSAMPLES}samples}
PARALLEL=${PARALLEL:-0}   # set to 1 to run algos in parallel after baselines

echo "Preparing baselines (g*, f*)…"
python main.py \
  --only-baselines \
  --n-samples "$NSAMPLES" \
  --label-idx "$LABEL_IDX" \
  --data-dir "$DATA_DIR" \
  --bound "$BOUND" \
  --domain "$DOMAIN" \
  --results-dir "$RESULTS_DIR"

if [[ "$PARALLEL" == "1" ]]; then
  echo "Running BiCS and FCBiO in parallel…"
  python main.py --algo bics \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --bics-iters "$BICS_ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo fcbio \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --fcbio-T "$FCBIO_T" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  wait
else
  echo "Running BiCS…"
  python main.py --algo bics \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --bics-iters "$BICS_ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running FCBiO…"
  python main.py --algo fcbio \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --fcbio-T "$FCBIO_T" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR"
fi

echo "Plotting comparison…"
python plot_compare.py --results-dir "$RESULTS_DIR" --algos BiCS FCBiO

echo "Done. Results in $RESULTS_DIR"
