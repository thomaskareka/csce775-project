import copy, torch, time, random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm


@register_algorithm("ramario")
class Reptile(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)

        self.episodes_per_task = config["episodes_per_task"]
        self.num_grad_steps = config["num_grad_steps"]
        self.inner_lr = config["inner_lr"]
        self.lr = config["lr"]

        self.model = self.model.to(self.device)

        self.epsilon_start = config["epsilon_start"]
        self.epsilon = self.epsilon_start
        self.min_epsilon = config["epsilon_min"]
        self.epsilon_steps = config["epsilon_steps"]

        self.save_every = config["save_every"]
        self.num_envs = config["num_envs"]

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
            q_values = model(obs_tensor)
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

    def reptile_update(self, task_model):
        with torch.no_grad():
            for param, task_param in zip(self.model.parameters(), task_model.parameters()):
                param.data.add_(self.lr * (task_param.data - param.data))

        self.param_updates += 1

    def train_step(self, task_model, obs, actions, rewards):
        task_optimizer = self.task_optimizer

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0
        if self.num_envs == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)

        loss = None
        for _ in range(self.num_grad_steps):
            q_values = task_model(obs_tensor)
            action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            loss = F.mse_loss(action_q_values, rewards_tensor)

            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()

        return loss.item() if loss is not None else None

    def train(self, total_tasks, callback):
        self.model.train()
        obs, _ = self.env.reset()

        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        pbar = tqdm(total=total_tasks, desc="training", unit="task")
        pbar.update(self.tasks_done)

        last_time = time.time()
        last_action_steps = self.action_steps
        last_save_step = self.tasks_done
        loss = None
        task_model = copy.deepcopy(self.model).to(self.device)

        reptile_t0, reptile_t1 = 0.0, 0.0

        while self.tasks_done < total_tasks:
            self.epsilon = self.update_epsilon()

            task_model.load_state_dict(self.model.state_dict())
            self.task_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)

            completed_episodes = 0

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
                    obs_batch = obs
                    actions_batch = actions
                else:
                    rewards_batch = np.array([rewards], dtype=np.float32)
                    dones = np.array([terminated or truncated], dtype=bool)
                    done_count = int(dones[0])
                    obs_batch = obs
                    actions_batch = np.array([actions])

                    if dones: self.env.reset()

                loss = self.train_step(task_model, obs_batch, actions_batch, rewards_batch)
                step_t3 = time.time()
                episode_rewards += rewards_batch
                completed_episodes += done_count
                self.episodes_done += done_count
                self.action_steps += self.num_envs

                for i, done in enumerate(dones):
                    if done:
                        episode_rewards[i] = 0.0

                obs = next_obs

                now = time.time()
                dt = now - last_time
                if dt > 1.0:
                    aps = (self.action_steps - last_action_steps) / dt
                    pbar.set_postfix({
                        "actions_per_sec": f"{aps:.1f}",
                        "loss": f"{loss:.4f}" if loss is not None else "—",
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