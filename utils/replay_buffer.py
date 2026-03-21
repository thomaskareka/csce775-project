import random, torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, terminal):
        self.buffer.append((
            state.astype(np.uint8, copy=False),
            action, reward,
            next_state.astype(np.uint8, copy=False),
            terminal
        ))
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.tensor(terminals, dtype=torch.float32, device=device)
        )
    
    def __len__(self):
        return len(self.buffer)