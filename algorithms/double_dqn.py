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
        self.num_envs = config["num_envs"]
        if self.num_envs > 1:
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.single_observation_space.shape)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.observation_space.shape)

        self.policy_net = self.model.to(self.device)
        self.target_net = copy.deepcopy(self.model).to(self.device)
        self.target_net.eval()
        
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(),
            lr=self.lr,
            momentum=self.grad_momentum,
            alpha=self.squared_momentum,
            eps=self.ms_grad,
            centered=True
        )
    

    def choose_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0

        if self.num_envs == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        if self.num_envs > 1:
            actions = greedy_actions.copy()
            explore_mask = np.random.rand(self.num_envs) < self.epsilon
            random_actions = np.array([
                self.env.single_action_space.sample()
                for _ in range(self.num_envs)
            ])
            actions[explore_mask] = random_actions[explore_mask]
            return actions

        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(greedy_actions[0])
    
    def update_epsilon(self):
        if self.action_steps >= self.epsilon_steps:
            return self.min_epsilon
        fraction = self.action_steps / self.epsilon_steps
        return 1.0 + fraction * (self.min_epsilon - 1)

    def train(self, total_steps, callback):
        obs, _ = self.env.reset()
        episode_rewards = np.zeros(self.num_envs)

        pbar = tqdm(total=total_steps, desc="training", unit="step")
        pbar.update(self.action_steps)
        start_time = time.time()
        last_time = start_time
        last_steps = 0
        last_save_step = 0
        loss = None
        train_t0, train_t1 = 0.0, 0.0

        while self.action_steps < total_steps:
            t0 = time.time()
            self.epsilon = self.update_epsilon()
            actions = self.choose_action(obs)
            t1 = time.time()

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            t2 = time.time()
            dones = np.logical_or(terminated, truncated)
            t3 = time.time()
            #paper clips to [-1,1]
            clipped_rewards = np.clip(rewards, -1, 1)

            self.replay_buffer.add_batch(obs, actions, clipped_rewards, next_obs, dones)

            obs = next_obs
            episode_rewards += rewards
            self.action_steps += self.num_envs
            self.emulator_frames += self.action_repeat * self.num_envs

            if self.emulator_frames >= self.replay_start_size:
                if self.param_updates == 0:
                    self.param_updates = self.action_steps // self.grad_update_freq
                desired_updates = self.action_steps // self.grad_update_freq
                updates_to_run = desired_updates - self.param_updates
                train_t0 = time.time()
                for _ in range(max(0, updates_to_run)):
                    loss = self.train_step()

                    if self.param_updates > 0 and self.param_updates % self.update_target_steps == 0:
                        self.update_target()
                train_t1 = time.time()

            if self.config["num_envs"] > 1:
                for i, done in enumerate(dones):
                    # print(f"reward: {episode_reward} frames: {self.emulator_frames} steps: {self.action_steps}")
                    if done:
                        episode_rewards[i] = 0.0
            else:
                if dones:
                    episode_rewards[0] = 0.0
                    obs, _ = self.env.reset()
            if callback and self.action_steps - last_save_step >= self.save_every:
                callback(f"checkpoint_{self.action_steps}.pt")
                last_save_step = self.action_steps
            
            pbar.update(self.num_envs)
            now = time.time()
            dt = now - last_time
            if dt > 1.0:
                sps = (self.action_steps - last_steps) / dt
                pbar.set_postfix({
                    "fps": f"{(sps * self.action_repeat):.1f}",
                    "epsilon": f"{self.epsilon:.3f}",
                    "loss": f"{loss:.4f}" if loss else "—",
                    "buf": len(self.replay_buffer),
                    "choose": f"{(t1 - t0)*1000:.1f}ms",
                    "env": f"{(t2 - t1)*1000:.1f}ms",
                    "train": f"{(train_t1 - train_t0)*1000:.1f}ms",
                })
                last_time = now
                last_steps = self.action_steps


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        q_values = self.policy_net(states / 255.0)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # use policy_net to select the best actions for next states
            policy_next_q = self.policy_net(next_states / 255.0)
            best_actions = policy_next_q.argmax(dim=1)
            
            # use target_net to evaluate those actions
            target_next_q = self.target_net(next_states / 255.0)
            max_next_q = target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
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