"""A simplistic implementation of the DDPG policy for residual RL."""

import torch
from torch import nn
import random


def init_critic_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Critic(nn.Module):
    """Simple possible critic network. Accepts the observations (proprioceptions) and actions"""
    def __init__(self, obs_shape, action_shape, cfg):
        """Build an ensemble of Q-functions for clipped target estimation."""
        super().__init__()
        self.cfg = cfg #num_experts = num_experts

        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], cfg.critic.hidden_dim),
            nn.LayerNorm(cfg.critic.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic.hidden_dim, cfg.critic.hidden_dim),
            nn.LayerNorm(cfg.critic.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.critic.hidden_dim, 1))
            for _ in range(cfg.critic.num_critics)])

        self.apply(init_critic_weights)

    def forward(self, obs, action):
        """Evaluate every critic on the provided state-action batch."""
        h_action = torch.cat([obs, action], dim=-1)
        h_action = h_action.float()
        q_out = [critic(h_action) for critic in self.critics]
        assert self.cfg.critic.num_critics == len(q_out)
        # Stack the list of num_critics tensors shape of [B, 1] into the [num_critics, B] tensor
        q_out_stack = torch.stack(q_out, dim=0).squeeze(-1)
        return q_out_stack


    def q_value(self, obs, act, aggregate_type='mean'):
        """
        Returns the Q-value for a given feature, property, and action.
        I.e., gets all Q-values, subsets them, and returns the min.
        Used for critic loss computation (uses configurable min_q_heads).
        """
        # Get the Q-values for all networks
        q_out = self.forward(obs, act)
        if aggregate_type == 'mean':
            # Convert list into a tensor to take mean.
            return q_out.mean(dim=0)
        elif aggregate_type == 'min':
            # Take min over random subset of heads (configurable via min_q_heads)
            num_critics = min(self.cfg.critic.num_sampled_critics, self.cfg.critic.num_critics)
            idx = torch.randperm(self.cfg.critic.num_critics, device=q_out.device)[:num_critics]
            return torch.min(q_out.index_select(0, idx), dim=0).values