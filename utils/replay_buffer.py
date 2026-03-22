import random, torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        n = len(states)
        indexes = (self.ptr + np.arange(n)) % self.capacity

        self.states[indexes] = states
        self.next_states[indexes] = next_states
        self.actions[indexes] = actions
        self.rewards[indexes] = rewards
        self.dones[indexes] = dones

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
    
    def sample(self, batch_size, device):
        indexes = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indexes]).to(device, dtype=torch.float32, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[indexes]).to(device, dtype=torch.float32, non_blocking=True)

        actions = torch.from_numpy(self.actions[indexes]).to(device)
        rewards = torch.from_numpy(self.rewards[indexes]).to(device)
        dones = torch.from_numpy(self.dones[indexes]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size