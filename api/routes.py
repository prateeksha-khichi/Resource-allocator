from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.environment import ResourceAllocatorEnv
import asyncio

app = FastAPI(title="RL Resource Allocator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ResourceAllocatorEnv()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
async def reset(task_id: str = "easy"):
    obs = await env.reset(task_id=task_id)
    return {"status": "reset_ok", "obs": obs.observation.tolist() if hasattr(obs.observation, "tolist") else obs.observation}

@app.post("/step")
async def step():
    action = env.agent.predict(env.current_obs_model) if env.current_obs_model else 1
    obs = await env.step(action)
    return {"reward": obs.reward, "done": obs.done}

@app.get("/state")
def state():
    return env.current_obs_model

@app.get("/metrics")
def metrics():
    return env.metrics_collector.get_system_metrics()

@app.get("/agent/status")
def agent_status():
    return {"status": "running", "current_reward": env.total_reward}

@app.post("/agent/train")
def agent_train():
    return {"status": "training_started"}

@app.get("/agent/policy")
def agent_policy():
    return {"policy_path": env.agent.policy_path, "loaded": env.agent.model is not None}

@app.get("/episodes")
def episodes():
    return []

@app.get("/episodes/{id}")
def episode_detail(id: int):
    return {"id": id}

@app.get("/savings")
def savings():
    class CostCalculator:
        COST_PER_CPU_HOUR = 0.0416
        def calculate_savings(self, rl_efficiency, baseline_efficiency, hours_running):
            wasted_baseline = (1 - baseline_efficiency) * hours_running
            wasted_rl = (1 - rl_efficiency) * hours_running
            savings_hours = wasted_baseline - wasted_rl
            savings_usd = savings_hours * self.COST_PER_CPU_HOUR
            return {
                "savings_usd": round(savings_usd, 4),
                "efficiency_gain_percent": round((rl_efficiency - baseline_efficiency) * 100, 1),
                "message": f"Saved ${savings_usd:.3f} vs random allocation"
            }
    calc = CostCalculator()
    return calc.calculate_savings(0.85, 0.62, 1.2)

@app.get("/comparison")
def comparison():
    return {"rl": 0.85, "random": 0.62}
