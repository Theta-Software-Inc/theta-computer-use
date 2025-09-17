from theta import Client
from theta.agents import OpenAIAgent, ClaudeAgent
import dotenv
import asyncio


async def main():
    await run_single_task("accounting-dilution", openai_agent=True, max_steps=10)

async def run_single_task(task_id, session_name="default", openai_agent=True, max_steps=20):
    client = Client(api_key=dotenv.get_key(".env", "THETA_API_KEY"))
    print("Closing any stale environments...")
    await client.close_all_environments()
    
    session = await client.create_session(session_name)
    print("Using session: ", session.name)

    print("Creating environment...")
    env = await session.create_environment(task_id)
    vnc_url = await env.get_vnc_url()
    print(f"Environment created with task: {env.task_id}, VNC available at {vnc_url}")
    input("Press Enter to continue...")
    
    agent = None
    if openai_agent:
        agent = OpenAIAgent(
            name="openai",
            api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
            model="computer-use-preview",
            screen_size=(1024, 768),
        )
    else:
        agent = ClaudeAgent(
            name="claude",
            api_key=dotenv.get_key(".env", "ANTHROPIC_API_KEY"),
            model="claude-4-sonnet-20250514",
            screen_size=(1024, 768),
        )

    print("Starting agent loop...")
    if env.materials:
        print("Setting materials...")
        agent.set_materials(env.materials)
    obs = env.current_obs
    for _ in range(max_steps):
        action, done = await agent.act(obs)
        
        if action is None or done:
            print("Agent finished or failed to find an action")
            break
            
        obs, reward, finished, info = await env.step(action)
        
        if finished:
            print("Environment indicates task is complete")
            break

    print("Evaluating environment...")
    run = await env.evaluate()
    print(f"Evaluation score: {run.score}")

    agent.trajectory_json(save_images=True, eval_score=run.score)
    agent.trajectory_html_viewer(eval_score=run.score)

    await env.close()
    print("Environment closed")

if __name__ == "__main__":
    asyncio.run(main())
