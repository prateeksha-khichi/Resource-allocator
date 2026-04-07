from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from pydantic import BaseModel

class DummyAction(BaseModel):
    container_id: str
    cpu_shares_delta: int
    memory_limit_delta_mb: float
    priority_change: int

class ResourceAllocatorAgent:
    def __init__(self):
        self.model = None
        self.policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "policies", "pretrained_ppo.zip")
    
    def load_pretrained(self):
        if os.path.exists(self.policy_path):
            self.model = PPO.load(self.policy_path)
        else:
            raise Exception("No pretrained policy found")
    
    def train(self, env, total_timesteps=50000):
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
        self.model.learn(total_timesteps=total_timesteps)
        os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)
        self.model.save(self.policy_path)
    
    def predict(self, observation):
        if self.model:
            action, _ = self.model.predict(observation, deterministic=True)
            discrete_act = int(action)
        else:
            discrete_act = 1 # stay
            
        return DummyAction(
            container_id="predicted_cid",
            cpu_shares_delta=256 if discrete_act == 2 else (-256 if discrete_act == 0 else 0),
            memory_limit_delta_mb=0,
            priority_change=0
        )
    
    def explain_decision(self, observation, action):
        return {
            "container": action.container_id,
            "current_cpu": observation.containers[0].cpu_percent if observation.containers else 0,
            "cpu_trend": "rising" if observation.containers and observation.containers[0].cpu_trend > 0 else "falling",
            "action_taken": f"CPU shares {'+' if action.cpu_shares_delta > 0 else ''}{action.cpu_shares_delta}",
            "reasoning": "Adjusted CPU to balance load",
            "confidence": 0.85
        }
