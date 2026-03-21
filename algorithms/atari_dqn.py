import copy, torch, random, time
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import deque

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm
from utils.replay_buffer import ReplayBuffer

@register_algorithm("atari_dqn")
class AtariDQN(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)

        self.gamma = config["gamma"]
        self.lr = config["lr"]

        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]

        self.epsilon_start = config["epsilon_start"]
        self.epsilon = self.epsilon_start
        self.min_epsilon = config["epsilon_min"]
        self.epsilon_steps = config["epsilon_steps"]

        self.update_target_steps = config["update_target_steps"]
        self.grad_update_freq = config["grad_update_freq"]
        self.action_repeat = config["action_repeat"]

        self.grad_momentum = config["grad_momentum"]
        self.squared_momentum = config["squared_momentum"]
        self.ms_grad = config["ms_grad"]

        self.replay_start_size = config["replay_start_size"]

        self.save_every = config["save_every"]

        self.emulator_frames = 0
        self.action_steps = 0
        self.param_updates = 0
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.policy_net = self.model
        self.target_net = copy.deepcopy(self.model).to(self.device)
        self.target_net.eval()
        
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.lr,
            momentum=self.grad_momentum,
            alpha=self.squared_momentum,
            eps=self.ms_grad
        )
    

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_tensor = torch.tensor(obs, dtype = torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(obs_tensor)
            return q_values.argmax(dim=1).item()
    
    def update_epsilon(self):
        if self.emulator_frames >= self.epsilon_steps:
            return self.min_epsilon
        fraction = self.action_steps / self.epsilon_steps
        return 1.0 + fraction * (self.min_epsilon - 1)

    def train(self, total_steps, callback):
        obs, _ = self.env.reset()

        episode_reward = 0.0
        pbar = tqdm(total=total_steps, desc="training", unit="step")
        pbar.update(self.action_steps)
        start_time = time.time()
        last_time = start_time
        last_steps = 0
        loss = None

        while self.action_steps < total_steps:
            self.epsilon = self.update_epsilon()
            action = self.choose_action(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            #paper clips to [-1,1]
            clipped_reward = float(np.sign(reward))

            self.replay_buffer.add(obs, action, clipped_reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            self.action_steps += 1
            self.emulator_frames += self.action_repeat

            if self.emulator_frames >= self.replay_start_size and self.action_steps % self.grad_update_freq == 0:
                loss = self.train_step()

                if self.param_updates > 0 and self.param_updates % self.update_target_steps == 0:
                    self.update_target()
            
            if done:
                # print(f"reward: {episode_reward} frames: {self.emulator_frames} steps: {self.action_steps}")
                obs, info = self.env.reset()
                episode_reward = 0.0
            
            if callback and self.action_steps % self.save_every == 0:
                callback()
            
            pbar.update(1)
            now = time.time()
            dt = now - last_time
            if dt > 1.0:
                sps = (self.action_steps - last_steps) / dt
                pbar.set_postfix({
                    "fps": f"{(sps * self.action_repeat):.1f}",
                    "epsilon": f"{self.epsilon:.3f}",
                    "loss": f"{loss:.4f}" if loss else "—",
                    "buf": len(self.replay_buffer),
                })
                last_time = now
                last_steps = self.action_steps


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        q_values = self.policy_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1).values
            target = rewards + self.gamma * max_next_q * (1.0 - dones)
        
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.param_updates += 1
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "target_net": self.target_net.state_dict(),
            "emulator_frames": self.emulator_frames,
            "action_steps": self.action_steps,
            "param_updates": self.param_updates,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state):
        print(state)

        self.optimizer.load_state_dict(state["optimizer"])
        self.target_net.load_state_dict(state["target_net"])

        self.emulator_frames = state["emulator_frames"]
        self.action_steps = state["action_steps"]
        self.param_updates = state["param_updates"]

        self.epsilon = state["epsilon"]