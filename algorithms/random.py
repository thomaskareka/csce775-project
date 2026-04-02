from algorithms import register_algorithm
from algorithms.base import BaseAlgorithm

@register_algorithm("random")
class Random(BaseAlgorithm):
    def __init__(self, model, env, device, config):
        super().__init__(model, env, device, config)
    
    def train(self, total_steps: int, callback=None, logger=None):
        obs, info = self.env.reset()
        
        episode_count = 0
        episode_reward = 0.0
        episode_length = 0

        for step in range(total_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                episode_count += 1
                if logger:
                    logger.log_episode_metric(episode_count - 1, {
                        "reward": episode_reward,
                        "length": episode_length
                    })
                episode_reward = 0.0
                episode_length = 0
                obs, info = self.env.reset()
            
            if step % 100 == 0:
                print(step, reward)
        
        # Return metrics for final results logging
        return {
            "last_loss": 0.0,
            "num_episodes": episode_count,
            "mean_return": 0.0,
            "std_return": 0.0,
            "mean_episode_length": 0.0
        }
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state):
        pass
