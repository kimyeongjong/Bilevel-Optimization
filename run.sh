#!/usr/bin/env bash
# Unified runner for BiCS/FCBiO/a-IRG. Computes baselines once, then runs algos
# sequentially or in parallel, and generates comparison plots.
set -euo pipefail

# Defaults (can override via env vars)
NSAMPLES=${NSAMPLES:-10000}
LABEL_IDX=${LABEL_IDX:-0}
BOUND=${BOUND:-50}
DOMAIN=${DOMAIN:-box}
ITERS=${ITERS:-1000}
EPS=${EPS:-0.1}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results/bd${BOUND}_${DOMAIN}_iters${ITERS}_eps${EPS}}
PARALLEL=${PARALLEL:-0}   # set to 1 to run algos in parallel after baselines

# a-IRG hyperparameters (see Corollary 3.5)
AIRG_GAMMA0=${AIRG_GAMMA0:-1.0}
AIRG_ETA0=${AIRG_ETA0:-1.0}
AIRG_B=${AIRG_B:-0.25}      # must be in (0, 0.5)
AIRG_R=${AIRG_R:-0.5}       # must be in [0, 1)

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
  echo "Running BiCS, FCBiO, and a-IRG in parallel…"
  python main.py --algo bics \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo fcbio \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo airg \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
    --airg-gamma0 "$AIRG_GAMMA0" --airg-eta0 "$AIRG_ETA0" --airg-b "$AIRG_B" --airg-r "$AIRG_R" &

  wait
else
  echo "Running BiCS…"
  python main.py --algo bics \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running FCBiO…"
  python main.py --algo fcbio \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running a-IRG…"
  python main.py --algo airg \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
    --airg-gamma0 "$AIRG_GAMMA0" --airg-eta0 "$AIRG_ETA0" --airg-b "$AIRG_B" --airg-r "$AIRG_R"
fi

echo "Plotting comparison…"
python plot_compare.py --results-dir "$RESULTS_DIR" --algos BiCS FCBiO aIRG

echo "Done. Results in $RESULTS_DIR"
