from theta import Client
from theta.agents import OpenAIAgent
import dotenv
import asyncio


async def main():
    client = Client(api_key=dotenv.get_key(".env", "THETA_API_KEY"))
    print("Closing any stale environments...")
    await client.close_all_environments()
    sessions = await client.get_sessions()
    if not sessions:
        session = await client.create_session("openai_cua")
    else:
        session = sessions[0]
    print("Using session: ", session.name)

    tasks = await client.get_tasks()
    print("Available tasks: ", tasks)

    print("Creating environment...")
    env = await session.create_environment("accounting-dilution")
    vnc_url = await env.get_vnc_url()
    print(f"Environment created with task: {env.task_id}, VNC available at {vnc_url}")
    input("Press Enter to continue...")
    agent = OpenAIAgent(
        name="openai",
        api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
        model="computer-use-preview",
        screen_size=(1024, 768),
    )

    print("Starting OpenAI CUA agent loop")
    obs = env.current_obs
    for _ in range(20):
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
