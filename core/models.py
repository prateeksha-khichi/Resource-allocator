from typing import List, Dict, Optional
from pydantic import BaseModel

class ContainerState(BaseModel):
    container_id: str
    name: str
    cpu_percent: float
    memory_percent: float
    cpu_trend: float        # positive=rising, negative=falling
    memory_trend: float
    priority: int           # 1=critical, 2=high, 3=normal, 4=low
    workload_type: str      # web/batch/ml/microservice
    current_cpu_shares: int # Docker cpu_shares value
    current_memory_limit_mb: float
    is_healthy: bool
    time_since_spike: float

class ResourceObservation(BaseModel):
    containers: List[ContainerState]
    total_cpu_available: float
    total_memory_available_mb: float
    system_load: float
    timestamp: float
    episode_step: int
    total_reward_so_far: float

class ResourceAction(BaseModel):
    container_id: str
    cpu_shares_delta: int       # -256 to +256
    memory_limit_delta_mb: float # -512 to +512
    priority_change: int        # -1, 0, +1

class ResourceReward(BaseModel):
    total: float
    performance_score: float
    efficiency_score: float
    stability_score: float
    prediction_bonus: float
    breakdown: Dict[str, float]
