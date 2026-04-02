import stable_retro

class BaseAlgorithm:
    def __init__(self, model, env, device, config):
        self.model = model
        self.env = env
        self.device = device
        self.config = config
    
    def train(self, total_steps: int, callback, logger=None):
        raise NotImplementedError

    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass