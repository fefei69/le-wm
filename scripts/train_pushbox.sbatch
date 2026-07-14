#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=lewm-pushbox
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/slurm-%j.out

cd /scratch/cw5167/workspace/le-wm
source .venv/bin/activate
python train.py --config-name=pushbox
