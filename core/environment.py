import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from core.models import ResourceObservation, ResourceAction, ContainerState
from core.metrics_collector import MetricsCollector
from core.workload_simulator import WorkloadSimulator
from core.reward import calculate_reward
from core.database import SessionLocal, Episode, Step

# We need to define env.agent so wait, environment uses agent?
# The task says: inference.py has `action = env.agent.predict(obs.observation)`
# So environment needs to hold reference to agent.

class ResourceAllocatorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.metrics_collector = MetricsCollector()
        self.simulator = WorkloadSimulator()
        
        # Max containers = 10 for simplicity in obs space
        self.max_containers = 10
        # Observation space: 10 containers * 5 features (cpu, memory, limit, shares, priority)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_containers * 5,), dtype=np.float32)
        # Action space: Discrete action per container, e.g. increase/decrease shares
        # For simplicity, 3 actions: 0=decrease, 1=stay, 2=increase
        self.action_space = spaces.Discrete(3)
        
        self.history = []
        self.episode_step = 0
        self.total_reward = 0.0
        self.current_obs_model = None
        
        from core.agent import ResourceAllocatorAgent
        self.agent = ResourceAllocatorAgent()
        # attempt to load pretrained
        try:
            self.agent.load_pretrained()
        except:
            pass

    def _get_obs(self):
        # build real metrics
        metrics = self.metrics_collector.get_container_metrics()
        
        containers = []
        for i, m in enumerate(metrics[:self.max_containers]):
            # Simulate workload trends using simulator for missing parts
            t = time.time()
            sim_cpu, sim_mem = self.simulator.ml_training_pattern(t)
            
            c = ContainerState(
                container_id=m["container_id"],
                name=m["name"],
                cpu_percent=m["cpu_percent"] if m["cpu_percent"] else sim_cpu*100,
                memory_percent=m["memory_percent"] if m["memory_percent"] else sim_mem*100,
                cpu_trend=0.0,
                memory_trend=0.0,
                priority=1 if i==0 else 3,
                workload_type="web",
                current_cpu_shares=1024,
                current_memory_limit_mb=512.0,
                is_healthy=True,
                time_since_spike=0.0
            )
            containers.append(c)
            
        sys_metrics = self.metrics_collector.get_system_metrics()
        obs = ResourceObservation(
            containers=containers,
            total_cpu_available=sys_metrics["cpu_percent"],
            total_memory_available_mb=sys_metrics["memory_available_mb"],
            system_load=sys_metrics["load_average"],
            timestamp=sys_metrics["timestamp"],
            episode_step=self.episode_step,
            total_reward_so_far=self.total_reward
        )
        self.current_obs_model = obs
        return obs

    def _obs_to_array(self, obs_model):
        arr = []
        for c in obs_model.containers:
            arr.extend([
                c.cpu_percent / 100.0,
                c.memory_percent / 100.0,
                c.current_cpu_shares / 2048.0,
                c.current_memory_limit_mb / 2048.0,
                c.priority / 5.0
            ])
        # pad if < 10 containers
        while len(arr) < self.max_containers * 5:
            arr.extend([0, 0, 0, 0, 0])
        return np.array(arr, dtype=np.float32)

    async def reset(self, seed=None):
        self.episode_step = 0
        self.total_reward = 0.0
        self.history = []
        obs_model = self._get_obs()
        
        class ObsWrapper:
            def __init__(self, m, env):
                self.observation = env._obs_to_array(m)
                self.done = False
                self.reward = 0.0
                
        return ObsWrapper(obs_model, self)
        
    def reset_sync(self, seed=None, options=None):
        # For gym compatibility
        self.episode_step = 0
        self.total_reward = 0.0
        self.history = []
        obs_model = self._get_obs()
        return self._obs_to_array(obs_model), {}

    async def step(self, action_dict):
        # expected action_dict like ResourceAction
        before_obs = self.current_obs_model
        
        # Apply action (simulated if no docker access)
        if isinstance(action_dict, ResourceAction):
            sim_action = action_dict
        else:
            # Map discrete action back to ResourceAction
            sim_action = ResourceAction(
                container_id=before_obs.containers[0].container_id if before_obs.containers else "unknown",
                cpu_shares_delta=256 if action_dict == 2 else (-256 if action_dict == 0 else 0),
                memory_limit_delta_mb=0,
                priority_change=0
            )

        # fetch new state
        self.episode_step += 1
        after_obs = self._get_obs()
        
        reward_obj = calculate_reward(before_obs, sim_action, after_obs)
        self.total_reward += reward_obj.total
        
        done = self.episode_step >= 100
        
        class ObsWrapper:
            def __init__(self, m, env, r, d):
                self.observation = env._obs_to_array(m)
                self.done = d
                self.reward = r
                self.actual_action = sim_action
        
        return ObsWrapper(after_obs, self, reward_obj.total, done)

    def step_sync(self, action):
        before_obs = self.current_obs_model
        sim_action = ResourceAction(
            container_id=before_obs.containers[0].container_id if before_obs.containers else "unknown",
            cpu_shares_delta=256 if action == 2 else (-256 if action == 0 else 0),
            memory_limit_delta_mb=0,
            priority_change=0
        )
        self.episode_step += 1
        after_obs = self._get_obs()
        reward_obj = calculate_reward(before_obs, sim_action, after_obs)
        self.total_reward += reward_obj.total
        done = self.episode_step >= 100
        
        return self._obs_to_array(after_obs), reward_obj.total, done, False, {}
