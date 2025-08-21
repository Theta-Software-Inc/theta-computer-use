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
    env = await session.create_environment(tasks[0])
    vnc_url = await env.get_vnc_url()
    print(f"Environment created with task: {env.task_id}, VNC available at {vnc_url}")

    agent = OpenAIAgent(
        name="cua",
        api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
        model="computer-use-preview",
        screen_size=(1024, 768),
    )

    print("Starting OpenAI CUA agent loop")
    obs = env.current_obs
    for step in range(5):
        print(f"Step {step + 1}: Agent is analyzing the screen...")
        action, done = agent.act(obs)
        
        if action is None or done:
            print("Agent finished or failed to find an action")
            break
            
        print(f"Agent taking action: {type(action).__name__}")
        obs, reward, finished, info = await env.step(action)
        
        if finished:
            print("Environment indicates task is complete")
            break

    print("Evaluating environment...")
    run = await env.evaluate()
    print(f"Evaluation score: {run.score}")

    print("Saving agent trajectory with evaluation score...")
    
    # Save trajectory with images as separate PNG files, including eval score
    trajectory_file = agent.trajectory_json(save_images=True, eval_score=run.score)
    print(f"✅ Trajectory JSON saved to: {trajectory_file}")
    
    # Generate HTML viewer for easy browsing, including eval score
    html_viewer = agent.trajectory_html_viewer(eval_score=run.score)
    print(f"🌐 HTML viewer generated: {html_viewer}")
    print("💡 Open the HTML file in your browser to view screenshots and trajectory details")
    
    await env.close()
    print("Environment closed")


if __name__ == "__main__":
    asyncio.run(main())
