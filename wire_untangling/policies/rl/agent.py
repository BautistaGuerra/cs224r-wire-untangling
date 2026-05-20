"""A simplistic implementation of the critic for the residual RL."""
import torch
from torch import nn
from torch.nn import MSELoss

import copy

from wire_untangling.policies.rl.critic import Critic
from wire_untangling.policies.rl.actor import RRLActor
from wire_untangling.policies.env_policies import DPFMModelPolicy


class TD3Agent(nn.Module):
    def __init__(self, obs_shape:tuple[int], action_shape:tuple[int], bc_actor:DPFMModelPolicy, cfg):
        super().__init__()
        self.cfg = cfg

        # self.bc_actor = bc_actor
        # create critics & actor
        self.critic = Critic(
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            cfg=self.cfg
        )
        self.actor = RRLActor(
            obs_shape=cfg.obs_shape, action_shape=cfg.action_shape, cfg=cfg)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.critic.lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor.lr)
        # A vanilla MSE loss for training critic Bellman loss
        self.critic_loss = MSELoss()

        self.to(self.cfg.device)

    def act_(self, obs: torch.Tensor, base_action: torch.Tensor, eval_mode:bool, stddev:float, use_target: bool, clip:float):
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, base_action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=clip)
        return action
    #
    # def act(self, obs: torch.Tensor, base_action: torch.Tensor, eval_mode=False, stddev=0.0, cpu=True) -> torch.Tensor:
    #     """This function takes tensor and returns actions in tensor"""
    #     assert not self.training
    #     assert not self.actor.training
    #     # Make a shallow copy of the observation dict
    #     obs = copy.copy(obs)
    #     # unsqueezed = self._maybe_unsqueeze_(obs)
    #
    #     action = self.act_(obs=obs, base_action=base_action, eval_mode=eval_mode, stddev=stddev, use_target=False, clip=None)
    #     action = action.detach()
    #     if cpu:
    #         action = action.cpu()
    #     return action

    def update_critic(self, obs:torch.Tensor, action:torch.Tensor, reward: torch.Tensor,
        discount: torch.Tensor, next_obs: torch.Tensor, next_action_base: torch.Tensor, stddev:float):
        with torch.no_grad():
            next_residual_action = self.act_(
                obs=next_obs,
                base_action=next_action_base,
                eval_mode=False,  # Sample the action
                stddev=stddev,
                clip=self.cfg.stddev_clip,
                use_target=True,
            )
            next_action = torch.clamp(next_action_base + next_residual_action, -1.0, 1.0)
            # For the purpose of updating critic, take the min aggregation of the target q-values
            target_q_min = self.critic_target.q_value(next_obs, next_action, aggregate_type='min')
            y = (reward + (discount * target_q_min)).detach()

        # Note: compute MSE loss on the critics. Original DDPG may use distributional loss
        # Here we get the K q-values from the K critic network as a Python list and stack them into a tensor
        q_predict_all = self.critic(obs, action)  # [K,B]
        # Compute MSE loss between predicted Q values and the fixed targets.
        # TODO(alexta): not sure about dimensions here
        loss = self.critic_loss(q_predict_all,  y.unsqueeze(0))
        # DDPG multiplies errors by importance weights. We don't have weights for now

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        # critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)
        self.critic_opt.step()

    def actor_loss_(self, obs: torch.Tensor, base_action: torch.Tensor,):
        # Act on the residual RL policy. Don't add the random noise, just get a simple mean of the action value
        # We will backprop into policy automatically here.
        action_pred: torch.Tensor = self.act_(obs=obs, base_action=base_action, eval_mode=False, stddev=0.0, use_target=False, clip=self.cfg.stddev_clip)

        # NOTE: actual ResFiT adds L2 penalty on action here

        # Combine the base action with the predicted action. Clamp to the valid range [-1, 1] to match environment.
        combined_action = torch.clamp(base_action + action_pred, -1.0, 1.0)

        # Note: we are different from the paper in taking MEAN, not min() here. We don't want to underestimate
        # Q value for the purpose of the policy optimization, so we are taking mean value here.
        q_values = self.critic.q_value(obs, combined_action, aggregate_type='mean')
        actor_loss = -q_values.mean()
        return actor_loss

    def update_actor(self, obs:torch.Tensor, action_base:torch.Tensor):
        actor_loss = self.actor_loss_(obs, action_base)
        # Standard packprop step
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)
        self.actor_opt.step()

    def update(self, batch, update_actor, stddev):
        for key in ['next_reward', 'gamma', 'next_done']:
            assert batch[key].dtype == torch.float32
            assert batch[key].dim() == 1 or batch[key].dim() == 2
            if batch[key].dim() == 2:
                assert batch[key].shape[1] == 1
                batch[key] = batch[key].squeeze(-1)
        obs: torch.Tensor = batch["obs"]
        action_base: torch.Tensor = batch["action_base"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch["next_reward"]
        discount: torch.Tensor = batch["gamma"]
        next_done: torch.Tensor = batch["next_done"]
        next_action_base: torch.Tensor = batch["next_action_base"]
        next_obs: torch.Tensor = batch["next_obs"]

        # Compute the effective discount, which is zeroed on terminal states.
        effective_discount = discount * (1-next_done)

        # Regular critic update step
        self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            discount=effective_discount,
            next_obs=next_obs,
            next_action_base=next_action_base,
            stddev=stddev
        )

        # Update the target critic network using EMA (Polyakov's) update
        self.update_target_ema(self.critic, self.critic_target, self.cfg.critic.tau)
        if not update_actor:
            return

        self.update_actor(obs, action_base)

    def update_target_ema(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # def step_lr_schedulers(self):
    #     """Step the learning rate schedulers for warmup."""
    #     if self.encoder_scheduler is not None:
    #         self.encoder_scheduler.step()
    #     if self.critic_scheduler is not None:
    #         self.critic_scheduler.step()
    #     if self.actor_scheduler is not None:
    #         self.actor_scheduler.step()
