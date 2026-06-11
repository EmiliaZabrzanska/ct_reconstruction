#!/bin/bash
#SBATCH -J tfpnp_run02
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/tfpnp_%j.out
#SBATCH -e logs/tfpnp_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/ct_reconstruction

mkdir -p logs results/learned

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/train_tfpnp.py \
    --output_dir results/learned/run_02_50pat_50ep \
    --n_patients 50 \
    --n_epochs 50 \
    --batch_size 8 \
    --n_grad_steps 4 \
    --m 5 \
    --N 6 \
    --eta 0.05 \
    --lr_policy 3e-5 \
    --lr_critic 1e-4 \
    --lr_pi2 1e-6 \
    --pi2_warmup 5 \
    --pi2_loss_scale 0.01 \
    --noise_std 0

echo "End: $(date)"