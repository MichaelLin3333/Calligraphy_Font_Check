#!/usr/bin/env bash
set -euo pipefail

# train.sh - launch training with sensible defaults for this project
# Usage: ./train.sh [DATA_DIR] [EPOCHS] [BATCH] [LR] [MAX_LR]

DATA_DIR="${1:-./chinese_fonts}"
EPOCHS="${2:-100}"
BATCH="${3:-32}"
LR="${4:-0.0005}"
MAX_LR="${5:-0.01}"
OPT="${6:-adamw}"
WD="${7:-1e-4}"
USE_ONECYCLE="${8:-true}"
USE_AMP="${9:-true}"
USE_TB="${10:-true}"
FREEZE="${11:-1}"
CHECKPOINT_FREQ="${12:-5}"

# Build argument string
ARGS=(--data-dir "${DATA_DIR}" --resume checkpoint_epoch_30.pth --epochs ${EPOCHS} --alpha 5.0 --batch-size ${BATCH} --lr ${LR} \
			--optimizer ${OPT} --weight-decay ${WD} \
			--checkpoint-freq ${CHECKPOINT_FREQ} --freeze-backbone-epochs ${FREEZE})

if [ "${USE_ONECYCLE}" = "true" ] || [ "${USE_ONECYCLE}" = "True" ]; then
	ARGS+=(--use-onecycle --max-lr ${MAX_LR})
fi

if [ "${USE_AMP}" = "true" ] || [ "${USE_AMP}" = "True" ]; then
	ARGS+=(--use-amp)
fi

if [ "${USE_TB}" = "true" ] || [ "${USE_TB}" = "True" ]; then
	ARGS+=(--use-tensorboard)
fi

# Optional: create a logs directory for TensorBoard if enabled
if [[ " ${ARGS[@]} " =~ "--use-tensorboard" ]]; then
	mkdir -p runs
	echo "TensorBoard logs will be written to ./runs"
fi

# Print & run
echo "Running: python -u train_model.py ${ARGS[*]}"
python -u train_model.py "${ARGS[@]}"

# Hint: run tensorboard in another shell to inspect training:
# tensorboard --logdir=runs --bind_all