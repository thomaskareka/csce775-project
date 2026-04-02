import copy, torch, random
import torch.optim as optim
import numpy as np
from collections import deque
from datetime import datetime

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm
from utils.replay_buffer import ReplayBuffer
from utils.metrics import EpisodeMetrics, ExperimentResults
from utils.metrics_aggregator import MetricsAggregator

@register_algorithm("dqn")
class DQN(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)

        self.gamma = config["gamma"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]

        self.epsilon_start = config["epsilon_start"]
        self.min_epsilon = config["epsilon_min"]
        self.epsilon_steps = config["epsilon_steps"]

        self.update_target_steps = config["update_target_steps"]

        self.replay_buffer = self.make_buffer()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.target_net = copy.deepcopy(self.model).to(self.device)
        self.target_net.eval()
        
        self.epsilon = self.min_epsilon  # For evaluation
    
    def choose_action(self, obs):
        """Choose action using epsilon-greedy policy for evaluation."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(obs_tensor)
                action = torch.argmax(q_values).item()
        
        return action
    
    def make_buffer(self):
        """Create replay buffer with appropriate observation shape."""
        return ReplayBuffer(self.buffer_size, self.env.observation_space.shape)
    
    def train(self, total_steps, callback, logger=None):
        obs, _ = self.env.reset()

        metrics_agg = MetricsAggregator(batch_size=10)
        episode_reward = 0.0
        episode_length = 0
        episode_idx = 0
        last_loss = 0.0
        
        # Mario-specific metrics per episode
        max_x_position = 0.0
        max_score = 0.0

        for step in range(total_steps):
            obs_tensor = torch.tensor(obs, dtype = torch.float32, device=self.device).unsqueeze(0)
            
            epsilon = max(self.min_epsilon, self.epsilon_start * (1 - step / self.epsilon_steps))
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(obs_tensor)
                    action = torch.argmax(q_values).item()
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if (reward > 0.0):
                print(info)
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Track Mario-specific metrics
            x_pos = ((info.get('xscrollHi', 0) << 8) | info.get('xscrollLo', 0))
            max_x_position = max(max_x_position, x_pos)
            max_score = max(max_score, info.get('score', 0.0))

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if done:
                # Calculate velocity (distance / time)
                velocity = max_x_position / episode_length if episode_length > 0 else 0.0
                
                # Level completion heuristic: terminated (not just truncated) with reasonable progress
                level_completed = (terminated and max_x_position > 100)
                
                ep = EpisodeMetrics(
                    episode_idx=episode_idx,
                    total_reward=episode_reward,
                    episode_length=episode_length,
                    final_epsilon=epsilon,
                    max_x_position=max_x_position,
                    max_score=max_score,
                    velocity=velocity,
                    level_completed=level_completed
                )
                metrics_agg.add_episode(ep)
                
                if logger:
                    logger.log_episode_metric(episode_idx, {
                        "reward": episode_reward,
                        "length": episode_length,
                        "max_x_position": max_x_position,
                        "max_score": max_score,
                        "velocity": velocity,
                        "level_completed": float(level_completed)
                    })
                
                episode_reward = 0.0
                episode_length = 0
                max_x_position = 0.0
                max_score = 0.0
                episode_idx += 1
                obs, _ = self.env.reset()
            
            if len(self.replay_buffer) < self.batch_size:
                continue

            last_loss = self.train_step()

            if logger and step % 100 == 0 and step > 0:
                logger.log_metrics(step, {"loss": last_loss, "epsilon": epsilon})

            if step % self.update_target_steps == 0:
                self.target_net.load_state_dict(self.model.state_dict())
            if step % 100 == 0:
                print(step, reward, action)
        
        # Flush any remaining metrics and return summary
        metrics_agg.flush()
        summary = metrics_agg.get_summary()
        
        # Return metrics for final results logging
        return {
            "last_loss": last_loss,
            "num_episodes": episode_idx,
            "mean_return": summary["mean_return"],
            "std_return": summary["std_return"],
            "mean_episode_length": summary["mean_episode_length"],
            "mean_max_x_position": summary.get("mean_max_x_position", 0.0),
            "mean_max_score": summary.get("mean_max_score", 0.0),
            "mean_velocity": summary.get("mean_velocity", 0.0),
            "success_rate": summary.get("success_rate", 0.0)
        }
    
    def train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        states = states.float()
        next_states = next_states.float()
        actions = actions.long()
        rewards = rewards.view(-1)
        dones = dones.view(-1).float()

        q_values = self.model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(1)[0]

            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "target_net": self.target_net.state_dict()
        }

    def load_state_dict(self, state):

        self.optimizer.load_state_dict(state["optimizer"])
        self.target_net.load_state_dict(state["target_net"])