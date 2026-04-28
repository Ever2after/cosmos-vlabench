#!/usr/bin/env bash

set -euo pipefail

export BASE_DATASETS_DIR=/NHNHOME/WORKSPACE/0426030040_A/data/vlabench-hdf5-if

CONDA_HOME=${CONDA_HOME:-/NHNHOME/WORKSPACE/0426030040_A/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-cosmos}

"$CONDA_HOME/bin/conda" run --no-capture-output -n "$CONDA_ENV_NAME" \
torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_vlabench_primitives_if_2" \
  trainer.grad_accum_iter=1