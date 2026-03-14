import copy, torch, random
import torch.optim as optim
import numpy as np
from collections import deque

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm
from utils.replay_buffer import ReplayBuffer

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

        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.target_net = copy.deepcopy(self.model).to(self.device)
        self.target_net.eval()
    
    def train(self, total_steps):
        obs, _ = self.env.reset()

        for step in range(total_steps):
            obs_tensor = torch.tensor(obs, dtype = torch.float32, device=self.device).unsqueeze(0)
            
            epsilon = max(self.min_epsilon, self.epsilon_start * (1 - step / self.epsilon_steps))
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = self.model(obs_tensor)
                    action = torch.argmax(q_values).item()
            
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
            if step % 100 == 0:
                print(step)
    
    def train_step(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(1)[0]

            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "target_net": self.target_net.state_dict()
        }

    def load_state_dict(self, state):

        self.optimizer.load_state_dict(state["optimizer"])
        self.target_net.load_state_dict(state["target_net"])