#!/bin/bash
#SBATCH -J tfpnp_ctdrunet_2
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

EXPERIMENT_NAME="run_12_pat_250_e80_ctdrunet"
DENOISER_PATH="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_ct/drunet_ct_best.pth"
FBPCONVNET_CKPT="/home/eaz21/rds/hpc-work/eaz21/results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth"

mkdir -p logs

if [ ! -f "$DENOISER_PATH" ]; then
    echo "ERROR: CT-DRUNet checkpoint not found at $DENOISER_PATH"
    exit 1
fi

echo "================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Denoiser:   $DENOISER_PATH (CT-specific)"
echo "Config:     250 train × 80 epochs — stabilised for CT-DRUNet"
echo "Start:      $(date)"
echo "================================================="

python scripts/train_tfpnp.py \
    --output_dir "results/learned/$EXPERIMENT_NAME" \
    --n_train 250 \
    --n_val 50 \
    --n_epochs 80 \
    --batch_size 4 \
    --n_grad_steps 2 \
    --m 5 \
    --N 6 \
    --eta 0.05 \
    --lr_policy 3e-5 \
    --lr_critic 3e-5 \
    --lr_pi2 1e-6 \
    --pi2_warmup 15 \
    --pi2_loss_scale 0.001 \
    --noise_std 0 \
    --buffer_size 5000 \
    --sigma_floor 4.0 \
    --sigma_ceil 12.0 \
    --mu_floor 7.0 \
    --mu_ceil 50.0 \
    --reward_type psnr \
    --reward_alpha 0.0 \
    --denoiser_path "$DENOISER_PATH"

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

python -u scripts/plot_training_curves.py --experiment_name "$EXPERIMENT_NAME"

python -u scripts/plot_checkpoint_comparison.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --fbpconvnet_ckpt "$FBPCONVNET_CKPT" \
        --denoiser_path "$DENOISER_PATH"

python -u scripts/plot_policy_behaviour.py --experiment_name "$EXPERIMENT_NAME"

python -u scripts/evaluate_run.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --n_test_subset 0 \
    --denoiser_path "$DENOISER_PATH" \
    --fbpconvnet_ckpt "$FBPCONVNET_CKPT"
    
python -u scripts/plot_reconstruction_gallery.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --fbpconvnet_ckpt "$FBPCONVNET_CKPT" \
        --denoiser_path "$DENOISER_PATH"

echo ""
echo "================================================="
echo "All outputs:"
echo "  Checkpoints:  /home/eaz21/rds/hpc-work/eaz21/results/learned/$EXPERIMENT_NAME/"
echo "  Metrics CSV:  /home/eaz21/rds/hpc-work/eaz21/results/metrics/$EXPERIMENT_NAME/"
echo "  Figures:      /home/eaz21/rds/hpc-work/eaz21/figures/$EXPERIMENT_NAME/"
echo "End: $(date)"
echo "================================================="
