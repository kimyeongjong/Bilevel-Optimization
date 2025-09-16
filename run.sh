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

# Prepare baselines unless they already exist for this RESULTS_DIR
BASE_FILE="$RESULTS_DIR/baselines.json"
if [[ -f "$BASE_FILE" ]]; then
  echo "Found existing baselines at $BASE_FILE — skipping computation."
else
  echo "Preparing baselines (g*, f*)…"
  python main.py \
    --only-baselines \
    --n-samples "$NSAMPLES" \
    --label-idx "$LABEL_IDX" \
    --data-dir "$DATA_DIR" \
    --bound "$BOUND" \
    --domain "$DOMAIN" \
    --results-dir "$RESULTS_DIR"
fi

if [[ "$PARALLEL" == "1" ]]; then
  echo "Running Bi-CS variants (RL,R,N,ER), FCBiO, a-IRG, IIBA in parallel…"
  python main.py --algo bics --bics-mode RL \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo bics --bics-mode R \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo bics --bics-mode N \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo bics --bics-mode ER --eps "$EPS" \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo fcbio \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  python main.py --algo airg \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
    --airg-gamma0 "$AIRG_GAMMA0" --airg-eta0 "$AIRG_ETA0" --airg-b "$AIRG_B" --airg-r "$AIRG_R" &

  python main.py --algo iiba \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &

  wait
else
  echo "Running Bi-CS-RL…"
  python main.py --algo bics --bics-mode RL \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running Bi-CS-R…"
  python main.py --algo bics --bics-mode R \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running Bi-CS-N…"
  python main.py --algo bics --bics-mode N \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"

  echo "Running Bi-CS-ER…"
  python main.py --algo bics --bics-mode ER --eps "$EPS" \
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

  echo "Running IIBA…"
  python main.py --algo iiba \
    --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
    --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"
fi

echo "Plotting comparison…"
python plot_compare.py --results-dir "$RESULTS_DIR" --algos "Bi-CS-RL" "Bi-CS-R" "Bi-CS-N" "Bi-CS-ER" "FC-BiO" "a-IRG" "IIBA"

echo "Done. Results in $RESULTS_DIR"
