import asyncio, os
from openai import OpenAI
from core.environment import ResourceAllocatorEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TASK_NAME = "resource_allocation"
BENCHMARK = "openenv-resource-allocator-v1"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 10.0
SUCCESS_SCORE_THRESHOLD = 0.6

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ResourceAllocatorEnv()
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        obs = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            action_obj = env.agent.predict(obs.observation)
            obs = await env.step(action_obj)
            
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=str(action_obj),
                    reward=reward, done=obs.done, error=None)
            if obs.done:
                break
        
        score = sum(rewards) / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    
    finally:
        log_end(success=success, steps=steps_taken,
               score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
