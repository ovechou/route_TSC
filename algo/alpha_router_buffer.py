import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrajectoryBuffer:
    """
    Per-CAV trajectory buffer aligned with original AlphaRouter design.
    Each CAV accumulates its own trajectory; at episode end, final reward
    is broadcast to all steps for REINFORCE training.
    """

    def __init__(self):
        self.trajectories = {}
        self.completed = []

    def new_trajectory(self, cav_id):
        self.trajectories[cav_id] = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'action_masks': [],
        }

    def store_step(self, cav_id, state, action, log_prob, value, action_mask=None):
        traj = self.trajectories[cav_id]
        traj['states'].append(state)
        traj['actions'].append(action)
        traj['log_probs'].append(log_prob)
        traj['values'].append(value)
        traj['action_masks'].append(action_mask)

    def finish_trajectory(self, cav_id, final_reward):
        if cav_id not in self.trajectories:
            return
        traj = self.trajectories.pop(cav_id)
        if len(traj['states']) == 0:
            return
        traj['final_reward'] = final_reward
        self.completed.append(traj)

    def discard_trajectory(self, cav_id):
        self.trajectories.pop(cav_id, None)

    @property
    def size(self):
        return sum(len(t['states']) for t in self.completed)

    @property
    def num_trajectories(self):
        return len(self.completed)

    def get_training_data(self):
        """
        Collate all completed trajectories into tensors.
        Reward is broadcast: every step in a trajectory gets the same final_reward.
        Returns dict of tensors or None if no data.
        """
        if len(self.completed) == 0:
            return None

        all_states = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_action_masks = []
        all_traj_log_prob_sums = []
        all_traj_rewards = []
        all_traj_values_mean = []

        for traj in self.completed:
            n = len(traj['states'])
            reward = traj['final_reward']

            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_log_probs.extend(traj['log_probs'])
            all_values.extend(traj['values'])
            all_rewards.extend([reward] * n)

            if traj['action_masks'][0] is not None:
                all_action_masks.extend(traj['action_masks'])

            all_traj_log_prob_sums.append(sum(traj['log_probs']))
            all_traj_rewards.append(reward)
            all_traj_values_mean.append(np.mean(traj['values']))

        states = torch.FloatTensor(np.array(all_states)).to(device)
        actions = torch.LongTensor(np.array(all_actions)).to(device)
        log_probs = torch.FloatTensor(np.array(all_log_probs)).to(device)
        values = torch.FloatTensor(np.array(all_values)).to(device)
        rewards = torch.FloatTensor(np.array(all_rewards)).to(device)

        action_masks = None
        if len(all_action_masks) > 0:
            action_masks = torch.FloatTensor(np.array(all_action_masks)).to(device)

        traj_log_prob_sums = torch.FloatTensor(all_traj_log_prob_sums).to(device)
        traj_rewards = torch.FloatTensor(all_traj_rewards).to(device)
        traj_values_mean = torch.FloatTensor(all_traj_values_mean).to(device)

        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'action_masks': action_masks,
            'traj_log_prob_sums': traj_log_prob_sums,
            'traj_rewards': traj_rewards,
            'traj_values_mean': traj_values_mean,
        }

    def clear(self):
        self.trajectories.clear()
        self.completed.clear()


class AlphaRouterTrainer:
    """
    REINFORCE + Value Baseline trainer aligned with original AlphaRouter.

    Training logic (from original pretrainer_module_pl.py):
      log_prob_sum = sum of log_probs per trajectory
      advantage = final_reward - value_baseline.detach()
      p_loss = advantage * log_prob_sum
      val_loss = MSE(value_predictions, final_reward_broadcast)
      loss = p_loss + val_loss_coef * val_loss + ent_coef * (-entropy)
    """

    def __init__(self, model, lr=1e-4, gamma=0.99,
                 val_loss_coef=0.5, ent_coef=0.01,
                 max_grad_norm=1.0, **kwargs):
        self.model = model
        self.gamma = gamma
        self.val_loss_coef = val_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.buffer = TrajectoryBuffer()
        self.training = False
        self.buf_cnt = 0

    def store_step(self, cav_id, state, action, log_prob, value, action_mask=None):
        self.buffer.store_step(cav_id, state, action, log_prob, value, action_mask)
        self.buf_cnt += 1

    def new_trajectory(self, cav_id):
        self.buffer.new_trajectory(cav_id)

    def finish_trajectory(self, cav_id, final_reward):
        self.buffer.finish_trajectory(cav_id, final_reward)

    def discard_trajectory(self, cav_id):
        self.buffer.discard_trajectory(cav_id)

    def learn(self, mini_batch_size=128):
        data = self.buffer.get_training_data()
        if data is None or data['states'].shape[0] < 2:
            return None

        self.training = True
        self.model.train()

        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        action_masks = data['action_masks']

        traj_log_prob_sums = data['traj_log_prob_sums']
        traj_rewards = data['traj_rewards']
        traj_values_mean = data['traj_values_mean']

        N = states.shape[0]
        all_values = torch.zeros(N, device=states.device)
        all_entropy = torch.zeros(N, device=states.device)

        for start in range(0, N, mini_batch_size):
            end = min(start + mini_batch_size, N)
            mb_states = states[start:end]
            mb_actions = actions[start:end]
            mb_masks = action_masks[start:end] if action_masks is not None else None

            probs, values = self.model(mb_states, action_mask=mb_masks)
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(probs)
            all_values[start:end] = values
            all_entropy[start:end] = dist.entropy()

        val_loss = nn.functional.mse_loss(all_values, rewards)

        if self.buffer.num_trajectories > 1:
            baseline = traj_rewards.mean()
        else:
            baseline = traj_values_mean

        advantage = traj_rewards - baseline.detach()
        p_loss = (advantage * traj_log_prob_sums).mean()

        entropy_bonus = -all_entropy.mean()

        loss = p_loss + self.val_loss_coef * val_loss + self.ent_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.clear()
        return loss.item()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
