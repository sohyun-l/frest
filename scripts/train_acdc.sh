#!/bin/bash
# Cityscapes -> ACDC source-free adaptation with FREST.
# Requires: DATA_DIR pointing to a folder containing ACDC/ (and pseudo_labels/),
# and ./pretrained_models/segformer.b5.1024x1024.city.160k.pth (source model).
export DATA_DIR=${DATA_DIR:?set DATA_DIR to the dataset root}
python -m tools.run fit --config configs/frest_acdc.yaml
