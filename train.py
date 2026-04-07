import os
from core.environment import ResourceAllocatorEnv
from core.agent import ResourceAllocatorAgent
from stable_baselines3.common.env_checker import check_env

# Custom Gym wrapper to match stable-baselines expectations
import gymnasium as gym

class SB3Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def reset(self, **kwargs):
        # ensure unpacking matches what sb3 expects
        return self.env.reset_sync(**kwargs)
    def step(self, action):
        return self.env.step_sync(action)

if __name__ == "__main__":
    base_env = ResourceAllocatorEnv()
    env = SB3Wrapper(base_env)
    
    agent = ResourceAllocatorAgent()
    agent.train(env, total_timesteps=50000)
