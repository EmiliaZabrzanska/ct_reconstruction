"""ADMM environment for TFPnP episode collection."""

import torch
import torch.nn.functional as F
import numpy as np

from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp


def _psnr(gt, recon):
    gt = gt.detach().cpu().numpy().squeeze()
    recon = recon.detach().cpu().numpy().squeeze()
    mse = np.mean((gt - recon)**2)
    data_range = gt.max() - gt.min()
    return float('inf') if mse < 1e-14 else 10 * np.log10(data_range**2 / mse)


def _ls_scale(gt, recon):
    alpha = (gt * recon).sum() / (recon * recon).sum()
    return recon * alpha


def _to_4d(t):
    return t.unsqueeze(0) if t.dim() == 3 else t


def collect_episode(policy, admm_step, replay_buffer, gt, op,
                    noise_std=None, noise_levels=(0.05, 0.075, 0.10),
                    m=5, N=6, eta=0.05):
    """
    Run one episode through the ADMM environment.
    
    If noise_std is None, randomly samples from noise_levels (paper convention).
    If noise_std is a float, uses that fixed value.
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
    x = _ls_scale(gt, x).clamp(min=0)
    z = x.clone()
    u = torch.zeros_like(x)
    noise_level = torch.tensor([noise_std], device=device)

    episode_rewards = []
    prev_psnr = _psnr(gt, x)

    for t in range(N):
        iter_frac = torch.tensor([t / N], device=device)

        with torch.no_grad():
            stop_logits, sigma_seq, mu_seq = policy(
                _to_4d(x), _to_4d(z), _to_4d(u), noise_level, iter_frac)
        stop_probs = F.softmax(stop_logits, dim=-1)
        stop = torch.bernoulli(stop_probs[:, 1]).bool().item()

        state = (x.detach().clone(), z.detach().clone(), u.detach().clone(),
                 noise_level.clone(), iter_frac.clone())

        with torch.no_grad():
            for i in range(m):
                x, z, u = admm_step(x, z, u, y,
                                     sigma=sigma_seq[0, i],
                                     mu=mu_seq[0, i])

        curr_psnr = _psnr(gt, x)
        reward = (curr_psnr - prev_psnr) - eta
        prev_psnr = curr_psnr
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