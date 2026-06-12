"""Loss functions for TFPnP training (eqs. 15–17 of Wei et al. 2022)."""

import torch
import torch.nn.functional as F

from ct_tfpnp.utils import to_4d


def compute_critic_loss(critic, target_critic, transitions, gamma=0.99):
    """TD loss for value network (eq. 15)."""
    device = next(critic.parameters()).device
    rewards, v_currs, v_nexts = [], [], []

    for t in transitions:
        s = tuple(to_4d(x, device) for x in t.state)
        ns = tuple(to_4d(x, device) for x in t.next_state)
        v_currs.append(critic(*s).squeeze())
        with torch.no_grad():
            v_nexts.append(target_critic(*ns).squeeze())
        rewards.append(t.reward.to(device).squeeze())

    v_curr = torch.stack(v_currs)
    v_next = torch.stack(v_nexts)
    r = torch.stack(rewards)

    td_target = r + gamma * v_next
    loss = 0.5 * F.mse_loss(v_curr, td_target)

    return loss, {
        'critic_loss': loss.item(),
        'mean_reward': r.mean().item(),
        'mean_v': v_curr.mean().item(),
    }


def compute_policy_loss_discrete(policy, critic, transitions, gamma=0.99):
    """REINFORCE loss for termination policy pi_1 (eq. 16)."""
    device = next(policy.parameters()).device
    advantages, log_probs_a1 = [], []

    for t in transitions:
        s = tuple(to_4d(x, device) for x in t.state)
        ns = tuple(to_4d(x, device) for x in t.next_state)

        with torch.no_grad():
            v_s = critic(*s).squeeze()
            v_ns = critic(*ns).squeeze()
        advantage = (t.reward.to(device).squeeze() + gamma * v_ns - v_s).detach()
        advantages.append(advantage)

        stop_logits, _, _ = policy(*s)
        log_probs = F.log_softmax(stop_logits, dim=-1)
        a1 = (F.softmax(t.action[0].to(device), dim=-1)[:, 1] > 0.5).long()
        log_probs_a1.append(log_probs[0, a1.item()])

    advantages = torch.stack(advantages)
    log_probs_a1 = torch.stack(log_probs_a1)
    loss = -(log_probs_a1 * advantages).mean()

    return loss, {
        'policy_loss_discrete': loss.item(),
        'mean_advantage': advantages.mean().item(),
    }


def compute_policy_loss_continuous(policy, critic, admm_step, transitions,
                                    m=5, N=6):
    """Model-based DDPG loss for continuous parameters pi_2 (eq. 17)."""
    device = next(policy.parameters()).device
    v_nexts = []

    for t in transitions:
        s = tuple(to_4d(x, device) for x in t.state)
        y = t.sinogram.to(device)

        _, sigma_seq, mu_seq = policy(*s)

        x_c = t.state[0].to(device)
        z_c = t.state[1].to(device)
        u_c = t.state[2].to(device)

        for i in range(sigma_seq.shape[1]):
            x_c, z_c, u_c = admm_step(x_c, z_c, u_c, y,
                                        sigma=sigma_seq[0, i],
                                        mu=mu_seq[0, i])

        next_iter = (t.state[4].to(device) + m / (N * m)).clamp(0, 1)
        v_next = critic(to_4d(x_c, device), to_4d(z_c, device),
                        to_4d(u_c, device),
                        t.state[3].to(device), next_iter)
        v_nexts.append(v_next.squeeze())

    loss = -torch.stack(v_nexts).mean()

    return loss, {
        'policy_loss_continuous': loss.item(),
        'mean_sigma': sigma_seq.mean().item(),
        'mean_mu': mu_seq.mean().item(),
    }