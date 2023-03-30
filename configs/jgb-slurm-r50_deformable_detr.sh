#!/usr/bin/env bash

set -x

EXP_DIR=dummy/E01-r50_deformable_detr-baseline-2
PY_ARGS=${@:1}

EXP_DIR
python -u main.py \
    --batch_size 4 \
    --epochs 20 \
    --output_dir ~/jgeob/OUTDIR/${EXP_DIR} \
    --isaid_path /nfs/projects/cv703/jazz-cvgroup-9/  \
    --detection_pretrained pretrained/r50_deformable_detr-checkpoint.pth
    ${PY_ARGS}
