from theta import Client
from theta.agents import OpenAIAgent, ClaudeAgent
import dotenv
import asyncio


async def main():
    tasks = ["accounting-dilution", "simulate-refund"] 
    await run_multiple_tasks(tasks, session_name="default", openai_agent=True, max_steps=10)

async def run_agent_loop(env, agent, max_steps: int):
    """Run one environment/agent loop to completion, evaluate, save trajectory, and close."""
    print(f"Starting agent loop for task {env.task_id}...")
    obs = env.current_obs
    for _ in range(max_steps):
        action, done = await agent.act(obs)
        if action is None or done:
            print(f"Agent finished or failed to find an action for task {env.task_id}")
            break
        obs, reward, finished, info = await env.step(action)
        if finished:
            print(f"Environment indicates task is complete for task {env.task_id}")
            break

    print(f"Evaluating environment for task {env.task_id}...")
    run = await env.evaluate()
    print(f"Evaluation score ({env.task_id}): {run.score}")

    agent.trajectory_json(save_images=True, eval_score=run.score)
    agent.trajectory_html_viewer(eval_score=run.score)

    await env.close()
    print(f"Environment closed for task {env.task_id}")


async def run_multiple_tasks(task_ids, session_name="default", openai_agent=True, max_steps=20):
    client = Client(api_key=dotenv.get_key(".env", "THETA_API_KEY"))
    print("Closing any stale environments...")
    await client.close_all_environments()

    session = await client.create_session(session_name)
    print("Using session: ", session.name)

    print("Creating environments...")
    envs = await session.create_environments(task_ids)
    vnc_urls = await asyncio.gather(*[env.get_vnc_url() for env in envs])
    print(
        f"Environments created with tasks: {', '.join([env.task_id for env in envs])}, "
        f"VNC available at {', '.join(vnc_urls)}"
    )
    input("Press Enter to continue...")

    agents = []
    for idx, env in enumerate(envs):
        if openai_agent:
            agents.append(
                OpenAIAgent(
                    name=f"openai_{idx+1}",
                    api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
                    model="computer-use-preview",
                    screen_size=(1024, 768),
                    enable_logging=False,
                )
            )
        else:
            agents.append(
                ClaudeAgent(
                    name=f"claude_{idx+1}",
                    api_key=dotenv.get_key(".env", "ANTHROPIC_API_KEY"),
                    model="claude-4-sonnet-20250514",
                    screen_size=(1024, 768),
                    enable_logging=False,
                )
            )

    tasks = [asyncio.create_task(run_agent_loop(env, agent, max_steps)) for env, agent in zip(envs, agents)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
