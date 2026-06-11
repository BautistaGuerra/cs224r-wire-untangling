"""Flow-matching behavior cloning policy for action chunks."""

import math
from typing import Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ---------------------------------------------------------------------------
# 1-D Temporal U-Net building blocks
# ---------------------------------------------------------------------------

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels
                              else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# 1-D Temporal U-Net (adapted from Chi et al., Diffusion Policy)
# ---------------------------------------------------------------------------

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 # diffusion_step_embed_dim=32,
                 # down_dims=(32, 64),
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups),
        ])

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(self, sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        sample = sample.moveaxis(-1, -2)  # (B,T,C) -> (B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond],
                                       dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)  # (B,C,T) -> (B,T,C)


class TemporalNoisePredictor(nn.Module):
    """Wraps ConditionalUnet1D behind a flat action-chunk interface.

    The flow schedule operates on flattened chunks ``(B, T * A)``. The U-Net
    itself receives temporal chunks ``(B, T, A)`` so action dimensions remain
    separate channels rather than being treated as one long scalar sequence.
    """

    def __init__(self, state_dim=4, pred_horizon=20, action_dim=1,
                 **unet_kwargs):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=state_dim,
            **unet_kwargs,
        )

    def forward(self, noisy_action, state, timestep):
        B = noisy_action.shape[0]
        x = noisy_action.view(B, self.pred_horizon, self.action_dim)
        out = self.unet(x, timestep, global_cond=state)
        return out.reshape(B, -1)


# ---------------------------------------------------------------------------
# Flow Matching schedule (conditional OT, Euler ODE sampler)
# ---------------------------------------------------------------------------

class FlowMatchingSchedule:
    """Conditional Optimal-Transport Flow Matching schedule.

    Implements the training-time interpolation and inference-time sampling
    for a flow matching policy.

    Args:
        action_dim: dimensionality of the flattened action chunk.
        device: torch device string.
        num_integration_steps: number of integration steps for sampling.
        clamp_sample: Optional final sample clamp. Disabled by default because
            checkpoint playback clips denormalized actions to env bounds.
    """

    def __init__(self, action_dim=1, device='cpu', num_integration_steps=20,
                 clamp_sample: bool = False):
        self.action_dim = action_dim
        self.device = device
        self.num_integration_steps = num_integration_steps
        self.clamp_sample = clamp_sample

    def interpolate(self, x1, t):
        """Build noisy sample x_t and the target velocity for training.

        Args:
            x1: clean action data, shape (B, action_dim).
            t: timesteps in [0, 1], shape (B,).

        Returns:
            (x_t, velocity) where both have shape (B, action_dim).
        """
        B, action_dim = x1.shape
        x0 = torch.randn(B, action_dim, device=x1.device)
        x_t = t[:, None]*x1 + (1-t[:, None])*x0
        velocity = x1 - x0
        return x_t, velocity

    @torch.no_grad()
    def sample(self, model, state, initial_noise: torch.Tensor | None = None):
        """Generate samples by integrating the learned velocity field.

        Args:
            model: the velocity network, callable as model(x, state, t).
            state: conditioning states, shape (B, state_dim).
            initial_noise: Optional starting point for the ODE. Passing zeros
                gives deterministic mean-like samples for unimodal BC policies.

        Returns:
            Sampled normalized action chunks, shape (B, action_dim).
        """
        B, _ = state.shape
        device = state.device
        if initial_noise is None:
            a = torch.randn(B, self.action_dim, device=device)
        else:
            a = initial_noise.to(device=device, dtype=state.dtype)
            if a.shape != (B, self.action_dim):
                raise ValueError(
                    f"initial_noise must have shape {(B, self.action_dim)}, got {tuple(a.shape)}"
                )
        dt = 1 / self.num_integration_steps
        t = torch.zeros(B, dtype=torch.float32, device=device)
        for _ in range(self.num_integration_steps):
            v = model(a, state, t)
            a = a + dt*v
            t = t + dt
        if self.clamp_sample:
            a = torch.clamp(a, -1.0, 1.0)
        return a


# ---------------------------------------------------------------------------
# Policy wrappers that bundle model + schedule
# ---------------------------------------------------------------------------

class FlowMatchingPolicy(nn.Module):
    """TemporalNoisePredictor + FlowMatchingSchedule bundled together."""

    def __init__(self, state_dim, pred_horizon, action_dim,
                 num_integration_steps, device='cuda'):
        """Implements the flow matching policy.
        Args:
            state_dim: dimensionality of the state
            pred_horizon: prediction horizon (aka chunk size for  action chunking)
            action_dim: dimensionality of the action
            num_integration_steps: number of integration steps for sampling.
        """
        super().__init__()
        self.model = TemporalNoisePredictor(
            state_dim=state_dim, pred_horizon=pred_horizon,
            action_dim=action_dim,
        )
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.schedule = FlowMatchingSchedule(
            action_dim=pred_horizon * action_dim,
            device=device,
            num_integration_steps=num_integration_steps,
        )

    @staticmethod
    def default_execute_steps(pred_horizon: int) -> int:
        """Default number of chunk actions to execute before re-planning."""
        return max(1, pred_horizon // 2)

    def forward(self, noisy_action, state, timestep):
        return self.model(noisy_action, state, timestep)


def mse_loss(policy, s_batch: torch.Tensor,
             a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the MSE regression loss for behavior cloning.

    Args:
        policy: BCPolicy network (callable: s_batch -> predicted actions).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    policy_batch = policy(s_batch)
    loss_value = torch.mean((policy_batch - a_batch)**2)
    return loss_value



def flow_matching_loss(policy, s_batch: torch.Tensor,
                       a_batch: torch.Tensor,
                       action_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the flow matching loss (MSE on velocity prediction).

    The policy (FlowMatchingPolicy) carries its own schedule.

    Args:
        policy: FlowMatchingPolicy (model + schedule).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).
        action_mask: optional mask with same shape as a_batch. Zero entries do
            not contribute to the MSE.

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    B, _ = s_batch.shape
    t = torch.rand(B, device=a_batch.device)
    a_t, velocity = policy.schedule.interpolate(a_batch, t)
    v_policy = policy(a_t, s_batch, t)
    squared_error = (v_policy - velocity)**2
    if action_mask is None:
        fm_loss = torch.mean(squared_error)
    else:
        if action_mask.shape != squared_error.shape:
            raise ValueError(
                f"action_mask shape {action_mask.shape} must match loss shape {squared_error.shape}"
            )
        action_mask = action_mask.to(dtype=squared_error.dtype)
        fm_loss = torch.sum(squared_error * action_mask) / torch.clamp(action_mask.sum(), min=1.0)
    return fm_loss
