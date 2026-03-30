import copy
import random
import torch
import torch.optim as optim

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm
from utils.replay_buffer import ReplayBuffer


@register_algorithm("double_dqn")
class DoubleDQN(BaseAlgorithm):
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

        state_shape = self.env.observation_space.shape
        self.replay_buffer = ReplayBuffer(self.buffer_size, state_shape)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.target_net = copy.deepcopy(self.model).to(self.device)
        self.target_net.eval()

    def train(self, total_steps, callback=None):
        obs, _ = self.env.reset()

        for step in range(total_steps):
            epsilon = max(self.min_epsilon, self.epsilon_start * (1 - step / self.epsilon_steps))
            
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(obs_tensor)
                    action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                obs, _ = self.env.reset()

            if len(self.replay_buffer) < self.batch_size:
                continue

            self.train_step()

            if step % self.update_target_steps == 0:
                self.target_net.load_state_dict(self.model.state_dict())

            if callback and step > 0 and step % self.config.get("save_every", 100000) == 0:
                callback()

    def train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        q_values = self.model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "target_net": self.target_net.state_dict(),
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.target_net.load_state_dict(state["target_net"])
