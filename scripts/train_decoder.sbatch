#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=lewm-decoder
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/slurm-%j.out

cd /scratch/cw5167/workspace/le-wm
source .venv/bin/activate
python train_decoder.py "$@"
