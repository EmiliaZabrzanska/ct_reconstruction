#!/bin/bash
#SBATCH -J full_eval_batch
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/full_eval_%j.out
#SBATCH -e logs/full_eval_%j.err

source ~/.bashrc
conda activate mphil_ct
cd ~/rds/hpc-work/eaz21

mkdir -p logs

FBPCONVNET_CKPT="/home/eaz21/rds/hpc-work/eaz21/results/learned/fbpconvnet_pat_250_e80/checkpoint_best_val.pth"

EXPERIMENTS=(
    "run_02_pat_100"
    "run_03_full"
    "run_04_pat_250_e80"
    "run_05_sigma_floor"
    "run_06_m3_N10"
    "run_07_ssim_reward"
)

echo "================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiments:   ${#EXPERIMENTS[@]}"
echo "FBPConvNet:    $FBPCONVNET_CKPT"
echo "Start:         $(date)"
echo "================================================="

for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "########################################################"
    echo "# Experiment: $EXPERIMENT_NAME"
    echo "# Started:    $(date)"
    echo "########################################################"

    echo ""
    echo "  [1/5] evaluate_run.py ..."
    python scripts/evaluate_run.py --experiment_name "$EXPERIMENT_NAME" --fbpconvnet_ckpt "$FBPCONVNET_CKPT"

    echo ""
    echo "  [2/5] plot_checkpoint_comparison.py ..."
    python scripts/plot_checkpoint_comparison.py --experiment_name "$EXPERIMENT_NAME" --fbpconvnet_ckpt "$FBPCONVNET_CKPT"

    echo ""
    echo "  [3/5] plot_policy_behaviour.py ..."
    python scripts/plot_policy_behaviour.py --experiment_name "$EXPERIMENT_NAME"

    echo ""
    echo "  [4/5] plot_reconstruction_gallery.py ..."
    python scripts/plot_reconstruction_gallery.py --experiment_name "$EXPERIMENT_NAME" --fbpconvnet_ckpt "$FBPCONVNET_CKPT"

    echo ""
    echo "  [5/5] plot_training_curves.py ..."
    python scripts/plot_training_curves.py --experiment_name "$EXPERIMENT_NAME"

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