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

GPUS=1
PORT="29501"
ROOTPATH=~/jgeob/rotb/

hostname
date
# conda activate cv703
cd ${ROOTPATH}
GPUS_PER_NODE=${GPUS} ./tools/run_dist_launch.sh ${GPUS} ${PORT} ./configs/slurm-r50_deformable_detr.sh