"""ADMM environment for TFPnP episode collection."""

import torch
import torch.nn.functional as F
import numpy as np

from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp
from ct_tfpnp.evaluation.metrics import psnr_np, ssim_np, haarpsi_np, ls_scale
from ct_tfpnp.utils import to_4d


def _compute_step_reward(gt, x_curr, prev_metrics, reward_type, alpha, eta):
    """
    Compute reward for one ADMM block transition. Returns (reward, new_metrics_dict).
    
    reward_type:
      'psnr'         — r = ΔPSNR − η  (paper default)
      'psnr_ssim'    — r = ΔPSNR + α·ΔSSIM − η
      'psnr_haarpsi' — r = ΔPSNR + α·ΔHaarPSI − η
    """
    curr_psnr = psnr_np(gt, x_curr)
    new_metrics = {'psnr': curr_psnr}

    if reward_type == 'psnr':
        reward = (curr_psnr - prev_metrics['psnr']) - eta
    elif reward_type == 'psnr_ssim':
        curr_ssim = ssim_np(gt, x_curr)
        new_metrics['ssim'] = curr_ssim
        reward = ((curr_psnr - prev_metrics['psnr']) +
                  alpha * (curr_ssim - prev_metrics['ssim'])) - eta
    elif reward_type == 'psnr_haarpsi':
        curr_haarpsi = haarpsi_np(gt, x_curr)
        new_metrics['haarpsi'] = curr_haarpsi
        reward = ((curr_psnr - prev_metrics['psnr']) +
                  alpha * (curr_haarpsi - prev_metrics['haarpsi'])) - eta
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    return reward, new_metrics


def collect_episode(policy, admm_step, replay_buffer, gt, op,
                    noise_std=None, noise_levels=(0.05, 0.075, 0.10),
                    m=5, N=6, eta=0.05,
                    reward_type='psnr', reward_alpha=0.0):
    """
    Run one episode through the ADMM environment.

    If noise_std is None, randomly samples from noise_levels (paper convention).
    If noise_std is a float, uses that fixed value.

    reward_type controls what drives the reward (see _compute_step_reward).
    Defaults to paper's PSNR-only — fully backward compatible.
    """
    device = gt.device

    # Randomly sample noise level if not specified (matches paper)
    if noise_std is None:
        noise_std = float(np.random.choice(noise_levels))

    # Generate noisy sinogram
    sino_clean = op.forward(gt)
    SCALE = sino_clean.max() / gt.max()
    sino_scaled = sino_clean / SCALE
    noise = noise_std * sino_scaled.std() * torch.randn_like(sino_scaled)
    sino_noisy = sino_scaled + noise
    y = sino_noisy * SCALE

    # FBP init
    x = lion_fbp(y, op)
    x = ls_scale(gt, x).clamp(min=0)
    z = x.clone()
    u = torch.zeros_like(x)
    noise_level = torch.tensor([noise_std], device=device)

    # Initialise prev_metrics for whichever reward type is chosen
    prev_metrics = {'psnr': psnr_np(gt, x)}
    if reward_type == 'psnr_ssim':
        prev_metrics['ssim'] = ssim_np(gt, x)
    elif reward_type == 'psnr_haarpsi':
        prev_metrics['haarpsi'] = haarpsi_np(gt, x)

    episode_rewards = []
    curr_psnr = prev_metrics['psnr']

    for t in range(N):
        iter_frac = torch.tensor([t / N], device=device)

        with torch.no_grad():
            stop_logits, sigma_seq, mu_seq = policy(
                to_4d(x), to_4d(z), to_4d(u), noise_level, iter_frac)
        stop_probs = F.softmax(stop_logits, dim=-1)
        stop = torch.bernoulli(stop_probs[:, 1]).bool().item()

        state = (x.detach().clone(), z.detach().clone(), u.detach().clone(),
                 noise_level.clone(), iter_frac.clone())

        with torch.no_grad():
            for i in range(m):
                x, z, u = admm_step(x, z, u, y,
                                     sigma=sigma_seq[0, i],
                                     mu=mu_seq[0, i])

        # Compute reward (handles all three reward types)
        reward, prev_metrics = _compute_step_reward(
            gt, x, prev_metrics, reward_type, reward_alpha, eta)
        curr_psnr = prev_metrics['psnr']
        episode_rewards.append(reward)

        # Store transition — move to CPU to save GPU memory
        next_iter = torch.tensor([(t + 1) / N], device=device)
        next_state = (x.detach().clone(), z.detach().clone(), u.detach().clone(),
                      noise_level.clone(), next_iter.clone())
        action = (stop_logits.detach(), sigma_seq.detach(), mu_seq.detach())

        def _cpu(tensors):
            return tuple(t.cpu() for t in tensors)

        replay_buffer.push(
            _cpu(state), _cpu(action),
            torch.tensor([reward]),
            _cpu(next_state),
            y.detach().cpu(),
        )

        if stop:
            break

    return {
        'rewards': episode_rewards,
        'final_psnr': curr_psnr,
        'n_decisions': t + 1,
        'n_admm_steps': (t + 1) * m,
    }