from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm

@register_algorithm("random")
class Random(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)
    
    def train(self, total_steps: int):
        obs, info = self.env.reset()

        for step in range(total_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                obs, info = self.env.reset()
            
            if step % 100 == 0:
                print(step, reward)
