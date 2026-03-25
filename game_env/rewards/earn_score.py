import gymnasium as gym
from . import register_reward

#stable retro's mario score is 1/10th of the displayed score, use a lower weight to prevent large reward spikes
@register_reward("earn_score")
class EarnScoreReward(gym.Wrapper):
    def __init__(self, env, weight = 0.1):
        super().__init__(env)
        self.weight = weight
        self.last_score = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        score = info.get("score", 0.0)
        if score > self.last_score:
            reward += (score - self.last_score) * self.weight
        self.last_score = score
        return obs, reward, terminated, truncated, info
