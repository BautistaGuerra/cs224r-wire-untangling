"""A simplistic implementation of the actor for the residual RL."""
import torch
from torch import nn
from wire_untangling.utils.normalizer import TruncatedNormal


def init_actor_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RRLActor(nn.Module):
    """A class that implements a residual RL actor."""

    def __init__(self,  obs_shape, action_shape, cfg):
        super(RRLActor, self).__init__()
        self.cfg = cfg

        self.policy = nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.p_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, action_shape[0]),
            nn.Tanh()
        )

        self.apply(init_actor_weights)


    def forward(self, obs, base_action, std):
        # Concat the observation and base action from the BC policy
        policy_input = torch.cat([obs, base_action], dim=-1)
        # Run the actor residual policy on it
        mu: torch.Tensor = self.policy(policy_input)

        # Scale the mean by action_scale
        # NOTE: std is already in environment action space (more interpretable)
        scaled_mu = mu * self.cfg.action_scale

        # Create distribution with scaled mean but environment-scale std
        action_dist = TruncatedNormal(scaled_mu, std)

        return action_dist  # noqa: RET504


