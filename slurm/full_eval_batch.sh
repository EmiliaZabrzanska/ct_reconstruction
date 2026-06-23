#!/bin/bash
#SBATCH -J full_eval_batch
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/full_eval_%j.out
#SBATCH -e logs/full_eval_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/eaz21

mkdir -p logs

FBPCONVNET_CKPT="/home/eaz21/rds/hpc-work/eaz21/results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth"
NATURAL_DRUNET="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_gray.pth"
CT_DRUNET="/home/eaz21/rds/hpc-work/eaz21/results/baselines/drunet_ct/drunet_ct_best.pth"

EXPERIMENTS=(
    "run_02_pat_100"
    "run_03_full"
    "run_04_pat_250_e80"
    "run_05_sigma_floor"
    "run_06_m3_N10"
    "run_07_ssim_reward"
    "run_08_haarpsi_reward"
    "run_09_ssim_only"
    "run_10_haarpsi_only"
    "run_11_pat_250_e80_ctdrunet"
)

echo "================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiments:   ${#EXPERIMENTS[@]}"
echo "FBPConvNet:    $FBPCONVNET_CKPT"
echo "Natural DRUNet:$NATURAL_DRUNET"
echo "CT DRUNet:     $CT_DRUNET"
echo "Start:         $(date)"
echo "================================================="

for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "########################################################"
    echo "# Experiment: $EXPERIMENT_NAME"
    echo "# Started:    $(date)"
    echo "########################################################"

    # Skip if not trained
    if [ ! -d "results/learned/$EXPERIMENT_NAME" ]; then
        echo "⚠  Skipping — results/learned/$EXPERIMENT_NAME does not exist"
        continue
    fi

    # Pick the denoiser this policy was trained with
    if [[ "$EXPERIMENT_NAME" == "run_11_pat_250_e80_ctdrunet" ]]; then
        DENOISER_PATH="$CT_DRUNET"
        echo "  Denoiser: CT-DRUNet (in-domain)"
    else
        DENOISER_PATH="$NATURAL_DRUNET"
        echo "  Denoiser: natural-image DRUNet"
    fi

    echo ""
    echo "  [1/5] evaluate_run.py ..."
    python -u scripts/evaluate_run.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --fbpconvnet_ckpt "$FBPCONVNET_CKPT" \
        --denoiser_path "$DENOISER_PATH" \
        || echo "⚠  evaluate_run.py failed for $EXPERIMENT_NAME"

    echo ""
    echo "  [2/5] plot_checkpoint_comparison.py ..."
    python -u scripts/plot_checkpoint_comparison.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --fbpconvnet_ckpt "$FBPCONVNET_CKPT" \
        --denoiser_path "$DENOISER_PATH" \
        || echo "⚠  plot_checkpoint_comparison.py failed for $EXPERIMENT_NAME"

    echo ""
    echo "  [3/5] plot_policy_behaviour.py ..."
    python -u scripts/plot_policy_behaviour.py --experiment_name "$EXPERIMENT_NAME" \
        || echo "⚠  plot_policy_behaviour.py failed for $EXPERIMENT_NAME"

    echo ""
    echo "  [4/5] plot_reconstruction_gallery.py ..."
    python -u scripts/plot_reconstruction_gallery.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --fbpconvnet_ckpt "$FBPCONVNET_CKPT" \
        --denoiser_path "$DENOISER_PATH" \
        || echo "⚠  plot_reconstruction_gallery.py failed for $EXPERIMENT_NAME"

    echo ""
    echo "  [5/5] plot_training_curves.py ..."
    python -u scripts/plot_training_curves.py --experiment_name "$EXPERIMENT_NAME" \
        || echo "⚠  plot_training_curves.py failed for $EXPERIMENT_NAME"

    echo ""
    echo "  Done $EXPERIMENT_NAME at $(date)"
done

echo ""
echo "================================================="
echo "All outputs:"
echo "  Checkpoints: /home/eaz21/rds/hpc-work/eaz21/results/learned/<exp>/"
echo "  Metrics CSV: /home/eaz21/rds/hpc-work/eaz21/results/metrics/<exp>/"
echo "  Figures:     /home/eaz21/rds/hpc-work/eaz21/figures/<exp>/"
echo "End: $(date)"
echo "================================================="
