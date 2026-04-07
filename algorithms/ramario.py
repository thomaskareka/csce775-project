import copy, torch, time, random
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm
from utils.replay_buffer import ReplayBuffer


@register_algorithm("ramario")
class Reptile(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)

        self.episodes_per_task = config["episodes_per_task"]
        self.num_grad_steps = config["num_grad_steps"]
        self.inner_lr = config["inner_lr"]
        self.lr = config["lr"]
        self.gamma = config["gamma"]

        self.batch_size = config["batch_size"]
        self.buffer_size = config["buffer_size"]

        self.model = self.model.to(self.device)

        self.save_every = config["save_every"]
        self.num_envs = config["num_envs"]

        if self.num_envs > 1:
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.single_observation_space.shape)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.observation_space.shape)

        self.tasks_done = 0
        self.action_steps = 0
        self.episodes_done = 0
        self.param_updates = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.task_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

    def choose_action(self, obs, model = None):
        if model is None:
            model = self.model
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device) / 255.0

        if self.num_envs == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            logits = model(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
        
        if self.num_envs > 1:
            return action.cpu().numpy()
        else:
            return int(action.item())
    
    def replay_buffer_update(self, task_model):
        if len(self.replay_buffer) < self.batch_size:
            return None

        last_loss = None
        for _ in range(self.num_grad_steps):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size, self.device
            )

            states = states / 255.0
            next_states = next_states / 255.0
            actions = actions.long().view(-1)
            rewards = rewards.float().view(-1)
            dones = dones.float().view(-1)

            logits = task_model(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            f_sa = logits.gather(1, actions.unsqueeze(1)).squeeze(1).detach()

            with torch.no_grad():
                next_logits = task_model(next_states)
                next_f_sa = next_logits.max(dim=1).values
                target = rewards + self.gamma * (1.0 - dones) * next_f_sa - f_sa

            loss = -(log_probs * target).mean()

            self.task_optimizer.zero_grad()
            loss.backward()
            self.task_optimizer.step()

            last_loss = loss.item()

        return last_loss
    
    def update_policy_gradient(self, task_model, obs, actions, rewards):
        if len(obs) == 0:
            return None
        
        obs_tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device) / 255.0
        actions_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device).view(-1)
        returns_tensor = self.calculate_returns(rewards)

        n = obs_tensor.shape[0]

        self.task_optimizer.zero_grad()
        total_loss = 0.0
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)

            obs_chunk = obs_tensor[start:end]
            actions_chunk = actions_tensor[start:end]
            returns_chunk = returns_tensor[start:end]

            logits = task_model(obs_chunk)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_chunk)

            chunk_loss = -(log_probs * returns_chunk).sum() / n
            chunk_loss.backward()
            total_loss += chunk_loss.item()

            del log_probs, dist, logits, obs_chunk, actions_chunk, returns_chunk
        self.task_optimizer.step()

        return total_loss
        
        
    def reptile_update(self, task_model):
        with torch.no_grad():
            for param, task_param in zip(self.model.parameters(), task_model.parameters()):
                param.data.add_(self.lr * (task_param.data - param.data))
        self.param_updates += 1
    
    def calculate_returns(self, rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        #normalization for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns

    def train(self, total_tasks, callback, logger=None):
        self.model.train()
        obs, _ = self.env.reset()

        pbar = tqdm(total=total_tasks, desc="training", unit="task")
        pbar.update(self.tasks_done)

        last_time = time.time()
        last_action_steps = self.action_steps
        last_save_step = self.tasks_done

        task_model = copy.deepcopy(self.model).to(self.device)
        replay_buffer_loss = None
        policy_gradient_loss = None
        reptile_t0, reptile_t1 = 0.0, 0.0

        while self.tasks_done < total_tasks:
            task_model.load_state_dict(self.model.state_dict())
            self.task_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)

            if self.num_envs > 1:
                self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.single_observation_space.shape)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size, self.env.observation_space.shape)

            completed_episodes = 0
            if self.num_envs > 1:
                trajectory_obs = [[] for _ in range(self.num_envs)]
                trajectory_actions = [[] for _ in range(self.num_envs)]
                trajectory_rewards = [[] for _ in range(self.num_envs)]
            else:
                trajectory_obs = []
                trajectory_actions = []
                trajectory_rewards = []

            while completed_episodes < self.episodes_per_task:
                step_t0 = time.time()
                actions = self.choose_action(obs, task_model)
                step_t1 = time.time()

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                step_t2 = time.time()

                if self.num_envs > 1:
                    rewards_batch = np.asarray(rewards, dtype=np.float32)
                    dones = np.logical_or(terminated, truncated)
                    done_count = int(dones.sum())

                    self.replay_buffer.add_batch(obs, actions, rewards_batch, next_obs, dones)
                    for i in range(self.num_envs):
                        trajectory_obs[i].append(obs[i])
                        trajectory_actions[i].append(actions[i])
                        trajectory_rewards[i].append(rewards[i])

                        if dones[i]:
                            policy_gradient_loss = self.update_policy_gradient(
                                task_model,
                                trajectory_obs[i],
                                trajectory_actions[i],
                                trajectory_rewards[i],
                            )
                            if policy_gradient_loss is not None:
                                policy_gradient_loss = float(policy_gradient_loss)

                            trajectory_obs[i] = []
                            trajectory_actions[i] = []
                            trajectory_rewards[i] = []
                else:
                    rewards_batch = np.array([rewards], dtype=np.float32)
                    dones = np.array([terminated or truncated], dtype=bool)
                    done_count = int(dones[0])

                    self.replay_buffer.add(np.asarray(obs), actions, rewards_batch[0], np.asarray(next_obs), dones[0])

                    trajectory_obs.append(obs)
                    trajectory_actions.append(actions)
                    trajectory_rewards.append(rewards)

                    if dones[0]:
                        policy_gradient_loss = self.update_policy_gradient(
                            task_model,
                            trajectory_obs,
                            trajectory_actions,
                            trajectory_rewards,
                        )
                        if policy_gradient_loss is not None:
                            policy_gradient_loss = float(policy_gradient_loss)

                        trajectory_obs = []
                        trajectory_actions = []
                        trajectory_rewards = []
                    
                replay_buffer_loss = self.replay_buffer_update(task_model)
                step_t3 = time.time()

                completed_episodes += done_count
                self.episodes_done += done_count
                self.action_steps += self.num_envs

                if self.num_envs == 1 and dones[0]:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs

                now = time.time()
                dt = now - last_time
                if dt > 1.0:
                    pbar.set_postfix({
                        "rb_l": f"{replay_buffer_loss:.4f}" if replay_buffer_loss is not None else "—",
                        "pg_l": f"{policy_gradient_loss:.4f}" if policy_gradient_loss is not None else "—",
                        "episodes": self.episodes_done,
                        "action_steps": self.action_steps,
                        "choose": f"{(step_t1 - step_t0)*1000:.1f}ms",
                        "env": f"{(step_t2 - step_t1)*1000:.1f}ms",
                        "train": f"{(step_t3 - step_t2)*1000:.1f}ms",
                        "reptile_ms": f"{(reptile_t1 - reptile_t0)*1000:.1f}",
                    })
                    last_time = now
                    last_action_steps = self.action_steps

            reptile_t0 = time.time()
            self.reptile_update(task_model)
            reptile_t1 = time.time()

            self.tasks_done += 1
            pbar.update(1)

            if callback and self.tasks_done - last_save_step >= self.save_every:
                callback()
                last_save_step = self.tasks_done
        
        pbar.close()
        
        # Return metrics for final results logging
        return {
            "last_loss": replay_buffer_loss if replay_buffer_loss else 0.0,
            "num_episodes": self.episodes_done,
            "mean_return": 0.0,  # Would need episode-level tracking
            "std_return": 0.0,
            "mean_episode_length": 0.0
        }

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "tasks_done": self.tasks_done,
            "action_steps": self.action_steps,
            "episodes_done": self.episodes_done,
            "param_updates": self.param_updates,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.model.load_state_dict(state["model"])

        self.tasks_done = state.get("tasks_done", 0)
        self.action_steps = state.get("action_steps", 0)
        self.episodes_done = state.get("episodes_done", 0)
        self.param_updates = state.get("param_updates", 0)