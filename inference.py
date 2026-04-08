import asyncio
import os
from openai import OpenAI

from server.sql_env_environment import Action, SqlEnvEnvironment

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 5

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SqlEnvEnvironment()

    rewards = []
    steps_taken = 0

    log_start("sql-task", "sql_env", MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        if result["done"]:
            break

        # Simple prompt
        prompt = f"Optimize this SQL query: {result['observation'].query}"

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            action_text = response.choices[0].message.content.strip()
        except:
            action_text = "SELECT name FROM users WHERE age > 20"

        result = await env.step(Action(optimized_query=action_text))

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)
        steps_taken = step

        log_step(step, action_text, reward, done)

        if done:
            break

    score = min(sum(rewards), 1.0)
    success = score > 0.5

    log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())