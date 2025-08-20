# Theta SDK

This SDK provides access to Theta's Computer Use Environments for agentic evals and post-training. Please contact founders@thetasoftware.ai for more info and support.

---

## 1. Prerequisites

- Python ≥ 3.9  
- Valid **Theta** API key

Create a `.env` file in the project root:

```env
THETA_API_KEY=your_theta_key
OPENAI_API_KEY=your_openai_key
```

## 2. Install


```bash
uv venv .venv && source .venv/bin/activate   
uv pip install -e .                          
```

## 3. Quick Start

```python
from theta import Client
from theta.agents import OpenAIAgent
import asyncio

async def main():
    # Initialize client
    client = Client()
    
     # Create a session for tracking runs
    session = await client.create_session(name="My Session")

    # Get available tasks
    tasks = await client.get_tasks()
    
    # Create an environment configured to a task
    env = await session.create_environment(task_id=tasks[0])
    
    # Get VNC URL for viewing the desktop
    vnc_url = await env.get_vnc_url()
    print(f"View at: {vnc_url}")
    
    # OpenAI CUA Agent
    agent = OpenAIAgent(
        name="cua",
        api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
        model="computer-use-preview",
        screen_size=(1024, 768),
    )

    # Agent Loop
    obs = env.current_obs # Set initial observation
    for step in range(5): # Define maximum number of steps
        action, done = agent.act(obs) # Agent takes an action based on the current observation

        if action is None or done:
            print("Agent finished or failed to find an action")
            break
            
        print(f"Agent taking action: {type(action).__name__}")
        obs, reward, finished, info = await env.step(action) # Agent receives new observation
        
        if finished:
            print("Environment indicates task is complete")
            break
    
    # Evaluate the task
    run = await env.evaluate()
    print(f"Score: {run.score}")

    # Clean up environment
    await env.close()

asyncio.run(main())
```

## 4. Parallel Environment Creation

The SDK now supports creating multiple environments in parallel:

```python
# Create multiple environments at once
envs = await session.create_environments(["task1", "task2", "task3"])
env1, env2, env3 = envs[0], envs[1], envs[2]

# List all running environments
await client.list_environments()

# Close all environments at once
await client.close_all_environments()
```

## 5. Trajectory Tracking

Store images, trajectory json, and html viewer in local "trajectories" directory:

```python
    
# Complete an eval run
run = env.evaluate()

# Save trajectory
trajectory_file = agent.trajectory_json(save_images=True, eval_score=run.score)
html_viewer = agent.trajectory_html_viewer(eval_score=run.score)

# Clean up environment
await env.close()
```
## 6. Full Example

See 'examples/' for agent-specific examples.
