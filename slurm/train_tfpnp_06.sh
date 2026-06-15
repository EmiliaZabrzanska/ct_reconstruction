#!/bin/bash
#SBATCH -J tfpnp_run06
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH -o logs/tfpnp_%j.out
#SBATCH -e logs/tfpnp_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/eaz21

EXPERIMENT_NAME="run_06_m3_N10"

mkdir -p logs

echo "================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Variant:    m=3, N=10 (paper-tested alternative, m*N=30 unchanged)"
echo "Start:      $(date)"
echo "================================================="

python scripts/train_tfpnp.py \
    --output_dir "results/learned/$EXPERIMENT_NAME" \
    --n_train 250 \
    --n_val 50 \
    --n_epochs 80 \
    --batch_size 8 \
    --n_grad_steps 4 \
    --m 3 \
    --N 10 \
    --eta 0.05 \
    --lr_policy 3e-5 \
    --lr_critic 1e-4 \
    --lr_pi2 1e-6 \
    --pi2_warmup 5 \
    --pi2_loss_scale 0.01 \
    --noise_std 0 \
    --buffer_size 5000 \
    --sigma_floor 1.0 \
    --sigma_ceil 5.0 \
    --mu_floor 10.0 \
    --mu_ceil 100.0 \
    --reward_type psnr \
    --reward_alpha 0.0 \
    --denoiser_path /home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed (exit $TRAIN_EXIT). Skipping post-processing."
    exit $TRAIN_EXIT
fi

# ── Post-training: figures + full evaluation ──────────────────────────
echo ""
echo "================================================="
echo "Training complete at $(date)."
echo "Running post-hoc evaluation on FULL test set..."
echo "================================================="

python scripts/plot_training_curves.py --experiment_name "$EXPERIMENT_NAME"
python scripts/plot_checkpoint_comparison.py --experiment_name "$EXPERIMENT_NAME"
python scripts/plot_policy_behaviour.py --experiment_name "$EXPERIMENT_NAME"
python scripts/evaluate_run.py --experiment_name "$EXPERIMENT_NAME" --n_test_subset 0

echo ""
echo "================================================="
echo "All outputs:"
echo "  Checkpoints:  results/learned/$EXPERIMENT_NAME/"
echo "  Metrics CSV:  results/metrics/$EXPERIMENT_NAME/"
echo "  Figures:      figures/$EXPERIMENT_NAME/"
echo "End: $(date)"
echo "================================================="