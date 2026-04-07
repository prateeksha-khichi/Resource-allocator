import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI
from core.environment import ResourceAllocatorEnv

# Environment variables as requested by OpenEnv Round 1
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or ""

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "rl-resource-allocator-v1"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 10.0

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    done_val = str(done).lower()
    error_val = str(error).lower() if error else "none"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def get_llm_action(client: OpenAI, obs_data: dict) -> int:
    """Uses LLM to decide the action (0=decrease, 1=stay, 2=increase)."""
    system_prompt = textwrap.dedent("""
        You are an RL Resource Allocator AI. 
        You manage CPU shares for Docker containers.
        Possible actions: 0 (decrease), 1 (stay), 2 (increase).
        Respond with ONLY the integer 0, 1, or 2.
    """).strip()
    
    user_prompt = f"Current container observations: {obs_data}. Choose action:"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=5,
            temperature=0.0
        )
        text = (completion.choices[0].message.content or "").strip()
        # extract integer
        if "0" in text: return 0
        if "2" in text: return 2
        return 1
    except Exception:
        return 1

async def run_task(client: OpenAI, env_instance: ResourceAllocatorEnv, task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        obs_wrapper = await env_instance.reset(task_id=task_id)
        
        for step in range(1, MAX_STEPS + 1):
            if obs_wrapper.done:
                break
            
            # USE LLM TO DECIDE ACTION
            action_int = await get_llm_action(client, obs_wrapper.observation.tolist())
            
            obs_wrapper = await env_instance.step(action_int)
            
            reward = obs_wrapper.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=str(action_int),
                    reward=reward, done=obs_wrapper.done, error=None)
            
            if obs_wrapper.done:
                break
        
        score = sum(rewards) / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.6
        
    except Exception as e:
        log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ResourceAllocatorEnv()
    
    for task in TASKS:
        await run_task(client, env, task)

if __name__ == "__main__":
    asyncio.run(main())
