#!/bin/bash
#SBATCH --job-name=exp-DfDt-1             # Job name
#SBATCH --output=/nfs/users/ext_cvgroup-9/jgeob/OUTDIR/slurm-logs/exp-DfDt1_%A_%N.txt # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=80G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --reservation=cv703         # for partition allocation
#SBATCH --partition=cv703           # for partition allocation

GPUS=2
PORT="29501"
ROOTPATH=~/jgeob/rotb/

hostname
date
# conda activate cv703
cd ${ROOTPATH}

## copy from run_dist_launch.sh
set -x

GPUS_PER_NODE=${GPUS}

RUN_COMMAND="./configs/r50_deformable_detr.sh"

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