#!/bin/bash
#SBATCH --job-name=exp-DfDt-2             # Job name
#SBATCH --output=/nfs/users/ext_cvgroup-9/jgeob/OUTDIR/slurm-logs/Ex2-DfDt1_%A_%N.txt # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=2                  # Run on a single CPU
#SBATCH --mem=80G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:2                # Number of GPUs (per node)
#SBATCH --time=24:00:00             # Specify the time needed for your experiment
#SBATCH --reservation=cv703         # for partition allocation
#SBATCH --partition=cv703           # for partition allocation

GPUS=2
PORT="29500"
EXP_DIR=/nfs/users/ext_cvgroup-9/jgeob/OUTDIR/E01-r50_deformable_detr-baseline-2
RESUMEPTH=/nfs/users/ext_cvgroup-9/jgeob/OUTDIR/E01-r50_deformable_detr-baseline-1/checkpoint.pth
ROOTPATH=~/jgeob/rotb/

hostname
date
# conda activate cv703
cd ${ROOTPATH}

## -----------------------------------------------------------------------------
set -x

GPUS_PER_NODE=${GPUS}

RUN_COMMAND="python -u main.py \
    --batch_size 4 \
    --epochs 55 \
    --output_dir ${EXP_DIR} \
    --resume ${RESUMEPTH} \
    --isaid_path /nfs/projects/cv703/jazz-cvgroup-9/ \
    --detection_pretrained pretrained/r50_deformable_detr-checkpoint.pth
    "

# RUN_COMMAND="./configs/r50_deformable_detr.sh"

if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-${PORT}}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}

echo "DONE TRAINING !!!"