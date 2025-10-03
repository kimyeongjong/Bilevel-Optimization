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
AIRG_ETA0=${AIRG_ETA0:-1.0} # must be > 0
AIRG_B=${AIRG_B:-0.1}       # must be in (0, 0.5)
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

  for MODE in RL R N ER; do
    OUT_JSON="$RESULTS_DIR/Bi-CS-$MODE.json"
    if [[ -f "$OUT_JSON" ]]; then
      echo "Found $OUT_JSON — skipping Bi-CS-$MODE."
    else
      if [[ "$MODE" == "ER" ]]; then EXTRA=(--eps "$EPS"); else EXTRA=(); fi
      python main.py --algo bics --bics-mode "$MODE" \
        --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
        --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
        "${EXTRA[@]}" &
    fi
  done

  # FCBiO
  if [[ -f "$RESULTS_DIR/FCBiO.json" ]]; then
    echo "Found $RESULTS_DIR/FCBiO.json — skipping FCBiO."
  else
    python main.py --algo fcbio \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR" &
  fi

  # a-IRG
  if [[ -f "$RESULTS_DIR/aIRG.json" ]]; then
    echo "Found $RESULTS_DIR/aIRG.json — skipping a-IRG."
  else
    python main.py --algo airg \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
      --airg-eta0 "$AIRG_ETA0" --airg-b "$AIRG_B" --airg-r "$AIRG_R" &
  fi

  # IIBA
  if [[ -f "$RESULTS_DIR/IIBA.json" ]]; then
    echo "Found $RESULTS_DIR/IIBA.json — skipping IIBA."
  else
    python main.py --algo iiba \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" &
  fi

  wait
else
  # Sequential execution with skip-if-exists guards
  for MODE in RL R N ER; do
    OUT_JSON="$RESULTS_DIR/Bi-CS-$MODE.json"
    if [[ -f "$OUT_JSON" ]]; then
      echo "Found $OUT_JSON — skipping Bi-CS-$MODE."
    else
      echo "Running Bi-CS-$MODE…"
      if [[ "$MODE" == "ER" ]]; then EXTRA=(--eps "$EPS"); else EXTRA=(); fi
      python main.py --algo bics --bics-mode "$MODE" \
        --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
        --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
        "${EXTRA[@]}"
    fi
  done

  if [[ -f "$RESULTS_DIR/FCBiO.json" ]]; then
    echo "Found $RESULTS_DIR/FCBiO.json — skipping FCBiO."
  else
    echo "Running FCBiO…"
    python main.py --algo fcbio \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --eps "$EPS" --seed "$SEED" --results-dir "$RESULTS_DIR"
  fi

  if [[ -f "$RESULTS_DIR/aIRG.json" ]]; then
    echo "Found $RESULTS_DIR/aIRG.json — skipping a-IRG."
  else
    echo "Running a-IRG…"
    python main.py --algo airg \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR" \
      --airg-eta0 "$AIRG_ETA0" --airg-b "$AIRG_B" --airg-r "$AIRG_R"
  fi

  if [[ -f "$RESULTS_DIR/IIBA.json" ]]; then
    echo "Found $RESULTS_DIR/IIBA.json — skipping IIBA."
  else
    echo "Running IIBA…"
    python main.py --algo iiba \
      --skip-optimum --n-samples "$NSAMPLES" --label-idx "$LABEL_IDX" --data-dir "$DATA_DIR" \
      --bound "$BOUND" --domain "$DOMAIN" --iters "$ITERS" --seed "$SEED" --results-dir "$RESULTS_DIR"
  fi
fi

echo "Plotting comparison…"
python plot_compare.py --results-dir "$RESULTS_DIR" --algos "Bi-CS-RL" "Bi-CS-R" "Bi-CS-N" "Bi-CS-ER" "FC-BiO" "a-IRG" "IIBA"

echo "Done. Results in $RESULTS_DIR"
