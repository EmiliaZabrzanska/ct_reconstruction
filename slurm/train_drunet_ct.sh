#!/bin/bash
#SBATCH -J drunet_ct
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o logs/drunet_ct_%j.out
#SBATCH -e logs/drunet_ct_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/eaz21

EXPERIMENT_NAME="drunet_ct"

mkdir -p logs

echo "================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: $EXPERIMENT_NAME (CT-specific DRUNet denoiser)"
echo "Start:      $(date)"
echo "================================================="

python scripts/train_drunet_ct.py \
    --output_dir "results/baselines/$EXPERIMENT_NAME" \
    --n_train 0 \
    --n_val 0 \
    --n_epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --sigma_max 50.0 \
    --val_sigmas "5,15,25" \
    --best_sigma 15.0

echo ""
echo "================================================="
echo "Done: $(date)"
echo "Drop-in checkpoint: results/baselines/$EXPERIMENT_NAME/drunet_ct_best.pth"
echo "================================================="