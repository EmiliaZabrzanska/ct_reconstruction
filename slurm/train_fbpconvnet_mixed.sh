#!/bin/bash
#SBATCH -J fbpconvnet_mixed_250_e80
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o logs/fbpconvnet_mixed_%j.out
#SBATCH -e logs/fbpconvnet_mixed_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/eaz21

EXPERIMENT_NAME="fbpconvnet_pat_250_e80_mixed"

mkdir -p logs

echo "================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Config:     250 train × 80 epochs (FBPConvNet, MIXED noise)"
echo "Start:      $(date)"
echo "================================================="

python scripts/train_fbpconvnet.py \
    --output_dir "results/learned/$EXPERIMENT_NAME" \
    --n_train 250 \
    --n_val 50 \
    --n_epochs 80 \
    --batch_size 4 \
    --lr 1e-4 \
    --noise_std 0.0

TRAIN_EXIT=$?

echo ""
echo "================================================="
echo "Done at $(date) (exit code: $TRAIN_EXIT)"
echo "  Checkpoint: /home/eaz21/rds/hpc-work/eaz21/results/learned/$EXPERIMENT_NAME/"
echo "================================================="

exit $TRAIN_EXIT