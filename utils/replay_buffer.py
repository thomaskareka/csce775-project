import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, terminal):
        self.buffer.append(
            (state, action, reward, next_state, terminal)
        )
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(*batch)
        return states, actions, rewards, next_states, terminals
    
    def __len__(self):
        return len(self.buffer)