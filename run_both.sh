#!/usr/bin/env bash
set -euo pipefail

# Defaults
NSAMPLES=${NSAMPLES:-10000}
LABEL_IDX=${LABEL_IDX:-0}
BOUND=${BOUND:-50}
BICS_ITERS=${BICS_ITERS:-1000}
FCBIO_T=${FCBIO_T:-1000}
EPS=${EPS:-0.1}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results/results_${NSAMPLES}samples}

echo "Running BiCS (with baselines)…"
python main.py \
  --algo bics \
  --n-samples "$NSAMPLES" \
  --label-idx "$LABEL_IDX" \
  --data-dir "$DATA_DIR" \
  --bound "$BOUND" \
  --bics-iters "$BICS_ITERS" \
  --seed "$SEED" \
  --results-dir "$RESULTS_DIR"

echo "Running FCBiO (reusing baselines)…"
python main.py \
  --algo fcbio \
  --n-samples "$NSAMPLES" \
  --label-idx "$LABEL_IDX" \
  --data-dir "$DATA_DIR" \
  --bound "$BOUND" \
  --fcbio-T "$FCBIO_T" \
  --eps "$EPS" \
  --seed "$SEED" \
  --results-dir "$RESULTS_DIR" \
  --skip-optimum

echo "Plotting comparison…"
python plot_compare.py --results-dir "$RESULTS_DIR" --algos BiCS FCBiO

echo "Done. Results in $RESULTS_DIR"

