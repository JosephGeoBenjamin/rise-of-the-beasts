#!/usr/bin/env bash

set -x

EXP_DIR=OUTDIR/dummy/r50_deformable_detr-baseline
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --detection_pretrained pretrained/r50_deformable_detr-checkpoint.pth
    ${PY_ARGS}