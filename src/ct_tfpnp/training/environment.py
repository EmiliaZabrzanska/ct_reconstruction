"""
ADMM environment for TFPnP episode collection.
"""

import torch
import torch.nn.functional as F
import numpy as np

from ct_tfpnp.ct_ops.fbp import fbp as lion_fbp, calibrate_to_data
from ct_tfpnp.evaluation.metrics import psnr_np, ssim_np, haarpsi_np
from ct_tfpnp.utils import to_4d, project_and_add_noise

_REWARD_METRICS = {
    "psnr":         ("psnr",),
    "psnr_ssim":    ("psnr", "ssim"),
    "psnr_haarpsi": ("psnr", "haarpsi"),
    "ssim":         ("psnr", "ssim"),     
    "haarpsi":      ("psnr", "haarpsi"),
}

# metric functions for each metric name
_METRIC_FNS = {"psnr": psnr_np, "ssim": ssim_np, "haarpsi": haarpsi_np}


def _measure(gt, x, reward_type):
    """
    Evaluate the metrics required by `reward_type`. Returns a dict.
    """
    if reward_type not in _REWARD_METRICS:
        raise ValueError(f"Unknown reward_type: {reward_type!r}")
    return {name: _METRIC_FNS[name](gt, x) for name in _REWARD_METRICS[reward_type]}


def _step_reward(prev, curr, reward_type, alpha, eta):
    """
    Reward for one ADMM block transition, from the metric dicts before and after.

        'psnr'         r = dPSNR - eta               
        'psnr_ssim'    r = dPSNR + alpha*dSSIM - eta
        'psnr_haarpsi' r = dPSNR + alpha*dHaarPSI - eta
        'ssim'         r = alpha*dSSIM - eta        
        'haarpsi'      r = alpha*dHaarPSI - eta

    `eta` is the per-step cost that makes early termination worthwhile.

    Args:
        prev, curr:  metric dicts from `_measure`, before and after the block.
        reward_type: see above.
        alpha:       weight on the perceptual term.
        eta:         per-step cost.

    Returns:
        The scalar reward as a Python float.
    """
    d = {k: curr[k] - prev[k] for k in curr}

    if reward_type == "psnr":
        r = d["psnr"]
    elif reward_type == "psnr_ssim":
        r = d["psnr"] + alpha * d["ssim"]
    elif reward_type == "psnr_haarpsi":
        r = d["psnr"] + alpha * d["haarpsi"]
    elif reward_type == "ssim":
        r = alpha * d["ssim"]
    elif reward_type == "haarpsi":
        r = alpha * d["haarpsi"]
    else:
        raise ValueError(f"Unknown reward_type: {reward_type!r}")

    return r - eta


def admm_init(y, op):
    """
    Initialise the ADMM state from the measurements alone.

    Args:
        y:  noisy sinogram, shape (1, n_angles, n_det).
        op: LION CT operator.

    Returns:
        (x, z, u) with x = z = calibrated FBP and u = 0.
    """
    # calibrated FBP initialisation 
    x = calibrate_to_data(lion_fbp(y, op), y, op).clamp(min=0)

    return x, x.clone(), torch.zeros_like(x)


def _to_cpu(tensors):
    """
    Detach a tuple of tensors and move to CPU (replay buffer lives on CPU).
    """
    return tuple(t.detach().cpu() for t in tensors)


def run_episode(policy,
                admm_step,
                gt,
                op,
                replay_buffer=None,
                noise_std=None,
                noise_levels=(0.05, 0.075, 0.10),
                m=5,
                N=6,
                eta=0.05,
                reward_type='psnr',
                reward_alpha=0.0,
                stochastic=True,
                seed=None):
    """
    Run one policy rollout through the ADMM environment.

    Args:
        policy:        ResNetActor_ADMM.
        admm_step:     ADMMStep.
        gt:            ground-truth image, shape (1, H, W).
        op:            LION CT operator.
        replay_buffer: if given, transitions are pushed to it.
        noise_std:     fixed fractional noise level.
        noise_levels:  candidate noise levels to sample from.
        m:             ADMM steps per decision.
        N:             maximum number of decisions.
        eta:           per-step cost in the reward.
        reward_type:   see `_step_reward`.
        reward_alpha:  weight on the perceptual term.
        stochastic:    True  — sample the stop action (training).
                       False — take the greedy stop action (validation / test).
        seed:          optional seed for the sinogram noise (local RNG only).

    Returns:
        dict with 'rewards', 'final_psnr', 'n_decisions', 'n_admm_steps',
        'noise_std' and 'x' (the final reconstruction).
    """
    device = gt.device

    # sample noise level
    if noise_std is None:
        noise_std = float(np.random.choice(noise_levels))

    # generate noisy sinogram
    y = project_and_add_noise(gt, op, noise_std, seed=seed)

    # initialise admm state
    x, z, u = admm_init(y, op)
    x0 = x.clone()

    # set policy to eval mode
    policy_was_training = policy.training
    policy.eval()

    # track metrics for reward computation
    noise_level = torch.tensor([noise_std], device=device)
    metrics = _measure(gt, x, reward_type)
    rewards, sigmas, mus = [], [], []
    t = 0

    # set fractional iteration number and state for the first decision
    for t in range(N):
        iter_frac = torch.tensor([t / N], device=device)
        state = (x.detach().clone(), z.detach().clone(), u.detach().clone(), noise_level.clone(), iter_frac.clone())

        # sample the policy to get the stop action and ADMM parameters
        with torch.no_grad():
            stop_logits, sigma_seq, mu_seq = policy(to_4d(x), to_4d(z), to_4d(u), noise_level, iter_frac)

            stop_prob = F.softmax(stop_logits, dim=-1)[0, 1]
            stop = bool(torch.bernoulli(stop_prob).item()) if stochastic \
                else bool(stop_prob.item() > 0.5)

            # record chosen parameters
            sigmas.extend(sigma_seq[0].tolist())
            mus.extend(mu_seq[0].tolist())

            # execute m ADMM steps 
            for i in range(m):
                x, z, u = admm_step(x, z, u, y, sigma=sigma_seq[0, i], mu=mu_seq[0, i])

        # Compute reward
        prev_metrics, metrics = metrics, _measure(gt, x, reward_type)
        reward = _step_reward(prev_metrics, metrics, reward_type, reward_alpha, eta)
        rewards.append(reward)

        done = stop or (t == N - 1)

        # store transition in replay buffer 
        if replay_buffer is not None:
            next_iter = torch.tensor([(t + 1) / N], device=device)
            next_state = (x.detach().clone(), z.detach().clone(), u.detach().clone(), noise_level.clone(), next_iter)
            action = (stop_logits.detach(), sigma_seq.detach(), mu_seq.detach())
            replay_buffer.push(
                _to_cpu(state),
                (bool(stop), *_to_cpu(action)),  
                torch.tensor([reward]),
                _to_cpu(next_state),
                y.detach().cpu(),
                gt.detach().cpu(),               
                bool(done),
            )

        if done:
            break
    
    # restore policy's original training/eval mode
    policy.train(policy_was_training)  

    return {
        "x": x.detach(),
        "x0": x0.detach(),
        "y": y.detach(),
        "sigmas": sigmas,
        "mus": mus,
        "rewards": rewards,
        "final_psnr": metrics["psnr"],
        "n_decisions": t + 1,
        "n_admm_steps": (t + 1) * m,
        "noise_std": noise_std,
    }

def collect_episode(policy, admm_step, replay_buffer, gt, op, **kwargs):
    """
    Backwards-compatible wrapper to keep old code working
    """
    return run_episode(policy, admm_step, gt, op, replay_buffer=replay_buffer, stochastic=True, **kwargs)