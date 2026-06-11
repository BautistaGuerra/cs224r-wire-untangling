"""A simplistic implementation of the critic for the residual RL."""
import torch
from torch import nn
from torch.nn import MSELoss
import numpy as np

import copy

from wire_untangling.policies.rl.critic import Critic
from wire_untangling.policies.rl.actor import RRLActor
from wire_untangling.utils.normalizer import maybe_unsqueeze, maybe_squeeze


class TD3Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # create critics & actor
        self.critic = Critic(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            cfg=self.cfg
        )
        self.actor = RRLActor(
            state_dim=cfg.state_dim, action_dim=cfg.action_dim, cfg=cfg)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.critic.lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.actor.lr)
        # A vanilla MSE loss for training critic Bellman loss
        self.critic_loss = MSELoss()

        self.to(self.cfg.device)

    # TODO(alexta): This is ugly as hell, having two copies of the same code. But should work for now.
    @staticmethod
    def get_combined_action_torch(base_action:torch.Tensor, residual_action:torch.Tensor) -> torch.Tensor:
        final_action = base_action + residual_action
        return final_action

    @staticmethod
    def get_combined_action_numpy(base_action:np.ndarray, residual_action:np.ndarray) -> np.ndarray:
        final_action = base_action + residual_action
        return final_action


    def act_(self, obs: torch.Tensor, base_action: torch.Tensor, eval_mode:bool, stddev:float, use_target: bool, clip:float):
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, base_action, stddev)
        if eval_mode:
            # This is an "inference mode" - eihter real inference, or the updating the actor.
            # Don't sample from the distribution with added mean, just deterministic policy
            action = dist.mean
        else:
            # This is a critic update. Sample with a bit of noise to promote critic smoothing.
            action = dist.sample(clip=clip)
        return action

    def act(self, obs: torch.Tensor, base_action: torch.Tensor, eval_mode=True, stddev=0.0, cpu=True, clip=None) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor - used during the policy roll-out (aka inference)"""
        assert not self.training
        assert not self.actor.training

        obs, was_unsqueezed = maybe_unsqueeze(obs)
        base_action, _ = maybe_unsqueeze(base_action)

        action = self.act_(obs=obs, base_action=base_action, eval_mode=eval_mode, stddev=stddev, use_target=False, clip=clip)
        action = maybe_squeeze(action, was_unsqueezed)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def update_critic(self, obs:torch.Tensor, action:torch.Tensor, reward: torch.Tensor,
                      discount: torch.Tensor, next_obs: torch.Tensor, next_action_base: torch.Tensor,
                      critic_smoothing_stddev:float):
        with torch.no_grad():
            next_residual_action = self.act_(
                obs=next_obs,
                base_action=next_action_base,
                # Sample the action to smooth the critic updates (i.e. critic should not get stuck
                # in the landscape minima).
                eval_mode=False,
                stddev=critic_smoothing_stddev,
                clip=self.cfg.critic_smoothing_stddev_clip,
                use_target=True,
            )
            # next_action = torch.clamp(next_action_base + next_residual_action, -1.0, 1.0)
            # Note: we believe that we should not clip the action by any means, as agent always work with normalized
            # action space.
            next_action = self.get_combined_action_torch(next_action_base, next_residual_action)
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
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.critic_grad_clip_norm)
        self.critic_opt.step()

        return {
            "train/critic_loss": loss.item(),
            "train/critic_qt": y.mean().item(),
            "train/critic_grad_norm": critic_grad_norm.item(),
            "_target_q": y.detach().cpu(),
        }

    def actor_loss_(self, obs: torch.Tensor, base_action: torch.Tensor,):
        # Act on the residual RL policy.
        # We will backprop into policy automatically here.
        action_pred: torch.Tensor = self.act_(
            obs=obs, base_action=base_action,
            # Don't add the random noise, just get a simple mean of the action value
            eval_mode=False, stddev=0.0, use_target=False,
            # Doesn't matter here.
            clip=self.cfg.critic_smoothing_stddev_clip)

        # NOTE: actual ResFiT adds L2 penalty on action here

        # Combine the base action with the predicted action.
        # Note: we believe that we should not clip the action by any means, as agent always work with normalized
        # action space.
        combined_action = self.get_combined_action_torch(base_action, action_pred)

        # Note: we are different from the paper in taking MEAN, not min() here. We don't want to underestimate
        # Q value for the purpose of the policy optimization, so we are taking mean value here.
        q_values = self.critic.q_value(obs, combined_action, aggregate_type='mean')
        actor_loss = -q_values.mean()
        return actor_loss, action_pred

    def update_actor(self, obs:torch.Tensor, action_base:torch.Tensor):
        actor_loss, action_pred = self.actor_loss_(obs, action_base)
        # Standard packprop step
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip_norm)
        self.actor_opt.step()

        return {
            "train/actor_loss": actor_loss.item(),
            "train/actor_grad_norm": actor_grad_norm.item(),
            "_actions": action_pred.detach().cpu(),
        }

    def update(self, batch, update_critic: bool, update_actor:bool, critic_smoothing_stddev:float):
        # Extract fields from TensorDict batch produced by MultiStepTransform.
        # Ref: train_residual_td3.py:614-618
        obs: torch.Tensor = batch["obs"]
        action_base: torch.Tensor = batch["action_base"]
        action: torch.Tensor = batch["action"]
        reward: torch.Tensor = batch[("next", "reward")]
        discount: torch.Tensor = batch["gamma"]
        nonterminal: torch.Tensor = batch["nonterminal"]
        next_obs: torch.Tensor = batch[("next", "obs")]
        next_action_base: torch.Tensor = batch[("next", "action_base")]

        metrics = None

        if update_critic:
            # Squeeze trailing dims if present
            if reward.dim() > 1:
                reward = reward.squeeze(-1)
            if discount.dim() > 1:
                discount = discount.squeeze(-1)

            # n-step discount already accounts for terminal states via nonterminal mask
            # Ref: train_residual_td3.py:618
            effective_discount = discount * nonterminal.float()

            # Regular critic update step. Do not add noise to the actions.
            metrics = self.update_critic(
                obs=obs,
                action=action,
                reward=reward,
                discount=effective_discount,
                next_obs=next_obs,
                next_action_base=next_action_base,
                critic_smoothing_stddev=critic_smoothing_stddev
            )

            # Update the target critic network using EMA (Polyakov's) update
            self.update_target_ema(self.critic, self.critic_target, self.cfg.tau)
        if update_actor:
            actor_metrics = self.update_actor(obs, action_base)
            if metrics is not None:
                metrics.update(actor_metrics)
            else:
                metrics = actor_metrics
            self.update_target_ema(self.actor, self.actor_target, self.cfg.tau)
        return metrics

    def update_target_ema(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

