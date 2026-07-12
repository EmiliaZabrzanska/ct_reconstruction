# slurm

CSD3 batch scripts (Ampere partition, `MPHIL-DIS-SL2-GPU`). 

Each activates `mphil_ct`, `cd`s to the work tree, trains, then runs the evaluations.

Submit with:

```bash
sbatch slurm/train_tfpnp_04.sh
```

## Scripts

| Script | Experiment |
|---|---|
| `train_tfpnp_02.sh` … `12.sh` | The numbered TFPnP runs (`run_02` … `run_12`). |
| `train_fbpconvnet.sh` | FBPConvNet baseline (fixed 5% noise). |
| `train_fbpconvnet_mixed.sh` | FBPConvNet on mixed {5, 7.5, 10}% noise. |
| `train_drunet_ct.sh` | CT-domain DRUNet denoiser. |
| `full_eval_batch.sh` | Re-run evaluation and figures for a list of experiments. |
