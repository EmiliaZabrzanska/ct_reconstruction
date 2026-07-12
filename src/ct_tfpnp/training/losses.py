"""
Loss functions for TFPnP training (eqs. 15–17 of Wei et al. 2022).
"""

import torch
import torch.nn.functional as F

from ct_tfpnp.evaluation.metrics import default_data_range, psnr
from ct_tfpnp.utils import to_4d

def _batch_states(states, device):
    """
    Stack a list of stored state tuples into one batched state.

    Args:
        states: list of B tuples stored as (1, H, W) images and shape-(1,) scalars.
        device: device to move the batch to.

    Returns:
        (x, z, u, noise_level, iter_frac) with shapes (B, 1, H, W) x3, (B,), (B,) ready to pass straight to policy/critic.
    """
    x = torch.cat([to_4d(s[0], device) for s in states], dim=0)
    z = torch.cat([to_4d(s[1], device) for s in states], dim=0)
    u = torch.cat([to_4d(s[2], device) for s in states], dim=0)
    noise = torch.cat([s[3].to(device).reshape(1) for s in states], dim=0)
    iter_frac = torch.cat([s[4].to(device).reshape(1) for s in states], dim=0)
    return x, z, u, noise, iter_frac

def compute_critic_loss(critic, target_critic, transitions, gamma=0.99):
    """
    TD loss for the value network (eq. 15), computed on a batch.

    Args:
        critic:        the value network being trained.
        target_critic: the EMA copy used for the bootstrap.
        transitions:   list of Transition namedtuples.
        gamma:         discount factor.

    Returns:
        (loss, info_dict)
    """
    # set device
    device = next(critic.parameters()).device

    # batch states and next states
    s = _batch_states([t.state for t in transitions], device)
    ns = _batch_states([t.next_state for t in transitions], device)

    # compute current and next state values
    v_curr = critic(*s).squeeze(-1)                                  
    with torch.no_grad():
        v_next = target_critic(*ns).squeeze(-1) 

    # compute reward and done flags
    r = torch.cat([t.reward.to(device).reshape(1) for t in transitions])
    done = torch.tensor([float(t.done) for t in transitions], device=device)

    # compute TD target and loss
    td_target = r + gamma * (1.0 - done) * v_next
    loss = 0.5 * F.mse_loss(v_curr, td_target)

    return loss, {
        "critic_loss": loss.item(),
        "mean_reward": r.mean().item(),
        "mean_v": v_curr.mean().item(),
    }


def compute_policy_loss_discrete(policy, critic, transitions, gamma=0.99):
    """
    REINFORCE loss for termination policy pi_1 (eq. 16).

    Args:
        policy:      ResNetActor_ADMM. Called ONCE on the whole batch.
        critic:      value network (baseline only; no gradient flows into it).
        transitions: list of Transition namedtuples.
        gamma:       discount factor.

    Returns:
        (loss, info_dict)
    """
    # set device amd batch size
    device = next(policy.parameters()).device
    B = len(transitions)

    # define batched states and next states
    s = _batch_states([t.state for t in transitions], device)
    ns = _batch_states([t.next_state for t in transitions], device)

    # Advantage r + gamma*(1-done)*V(s') - V(s), treated as a constant
    with torch.no_grad():

        # calculate V(s) and V(s') for the batch
        v_s = critic(*s).squeeze(-1)
        v_ns = critic(*ns).squeeze(-1)

        # compute reward and done flags
        r = torch.cat([t.reward.to(device).reshape(1) for t in transitions])
        done = torch.tensor([float(t.done) for t in transitions], device=device)

        # compute advantages for the batch
        advantages = r + gamma * (1.0 - done) * v_ns - v_s        

    # compute logits for the batch
    stop_logits, _, _ = policy(*s)                                  
    log_p = F.log_softmax(stop_logits, dim=-1)

    # Log-probability of action sampled at collection time
    actions = torch.tensor([int(t.action[0]) for t in transitions], device=device)
    log_probs = log_p[torch.arange(B, device=device), actions]        # (B,)

    loss = -(log_probs * advantages).mean()

    # compute entropy for logging
    entropy = -(log_p.exp() * log_p).sum(dim=-1).mean()

    return loss, {
        "policy_loss_discrete": loss.item(),
        "mean_advantage": advantages.mean().item(),
        "pi1_entropy": entropy.item(),
        "stop_rate": actions.float().mean().item(),
    }

def compute_policy_loss_continuous(policy, 
                                   critic, 
                                   admm_step, 
                                   transitions,
                                   m=5, 
                                   N=6, 
                                   gamma=0.99, 
                                   eta=0.05,
                                   include_reward=True):
    """
    Model-based deterministic policy gradient for the continuous parameters pi_2
    (eq. 17).
    
    Args:
        policy:         ResNetActor_ADMM.
        critic:         value network.
        admm_step:      ADMMStep — must be differentiable w.r.t. sigma and mu.
        transitions:    list of Transition namedtuples.
        m:              ADMM steps per decision (documentation only).
        N:              max decisions per episode; sets the iter_frac increment.
        gamma:          discount factor.
        eta:            per-step cost, matching the environment's reward.
        include_reward: include the differentiable r(s, a) term (paper's eq. 17).

    Returns:
        (loss, info_dict)
    """
    # set device
    device = next(policy.parameters()).device

    # define batch states
    s = _batch_states([t.state for t in transitions], device)

    # compute sigma_seq and mu_seq for the batch (no gradient flows into policy)
    _, sigma_seq, mu_seq = policy(*s)                                

    # create lists for the post-block states and rewards
    xs, zs, us, rewards = [], [], [], []

    # run the ADMM block for each transition in the batch
    for b, t in enumerate(transitions):
        y = t.sinogram.to(device)
        x_c, z_c, u_c = (t.state[i].to(device) for i in range(3))

        for i in range(sigma_seq.shape[1]):
            x_c, z_c, u_c = admm_step(x_c, z_c, u_c, y, sigma=sigma_seq[b, i], mu=mu_seq[b, i])

        xs.append(to_4d(x_c, device))
        zs.append(to_4d(z_c, device))
        us.append(to_4d(u_c, device))

        # compute reward if requested
        if include_reward:
            gt = t.gt.to(device)
            dr = default_data_range(gt)
            rewards.append(psnr(x_c, gt, dr) - psnr(t.state[0].to(device), gt, dr) - eta)

    # compute next state values for the batch
    next_iter = (s[4] + 1.0 / N).clamp(0, 1)
    v_next = critic(torch.cat(xs), torch.cat(zs), torch.cat(us),
                    s[3], next_iter).squeeze(-1)     

    # stack rewards or compute objective for the batch 
    if include_reward:
        r = torch.stack(rewards)                                      # (B,)
        objective = r + gamma * v_next                                     
    else:
        objective = v_next

    # compute loss for the batch
    loss = -objective.mean()

    # compute info dict for logging
    info = {
        "policy_loss_continuous": loss.item(),
        "mean_sigma": sigma_seq.mean().item(),
        "mean_mu": mu_seq.mean().item(),
    }
    if include_reward:
        info["pi2_mean_reward"] = r.mean().item()
    
    return loss, info