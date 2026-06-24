#!/bin/bash
# Evaluate a FREST checkpoint on the ACDC validation set (mIoU).
# Alignment weights are not needed at test time (encoder-only inference), so we
# skip the UAWarpC download; weights are restored from the checkpoint anyway.
export DATA_DIR=${DATA_DIR:?set DATA_DIR to the dataset root}
CKPT=${1:-checkpoints/frest_acdc.ckpt}
python -m tools.run test --config configs/frest_acdc.yaml \
  --ckpt_path "$CKPT" \
  --model.init_args.alignment_head.init_args.pretrained null
