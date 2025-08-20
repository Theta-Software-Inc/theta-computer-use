from typing import Optional
import uuid
from .requests import make_request
from .settings import settings
from .models import Observation, Run, EnvStatus, Action


class Environment:
    """
    Environment class for Theta Computer Use Environments
    """
    def __init__(
        self, 
        env_id: uuid.UUID,
        session: str,
        task_id: str,
        status: EnvStatus,
        current_obs: Optional[Observation] = None,
        vnc_url: Optional[str] = None
    ) -> None:
        """
        Initialize an environment

        Args:
            env_id: The environment UUID
            session: The session name this environment belongs to
            task_id: The task ID for this environment
            status: The environment status
            current_obs: The current observation
            vnc_url: The VNC URL for the environment
        """
        self.env_id = env_id
        self.session = session
        self.task_id = task_id
        self.status = status
        self.current_obs = current_obs
        self.vnc_url = vnc_url

    async def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute an action in the environment

        Args:
            action: The action to execute (ClickAction, ScrollAction, etc.)

        Returns:
            Observation: The observation of the environment
            float: The reward for the action
            bool: Whether the environment is done
            dict: Miscellaneous environment info
        """
        url = f"{settings.base_url}/environments/{self.env_id}/step"
        
        response = await make_request(
            url, 
            "PUT", 
            settings.api_key, 
            action.model_dump()
        )
        
        self.current_obs = Observation(**response["observation"])
        return self.current_obs, response["reward"], response["done"], response["info"]
    
    async def get_vnc_url(self) -> str:
        """
        Get the VNC URL for the environment

        Returns:
            str: The VNC URL for remote access to the environment
        """
        url = f"{settings.base_url}/environments/{self.env_id}/vnc"
        self.vnc_url = await make_request(url, "GET", settings.api_key)
        return self.vnc_url

    async def evaluate(self) -> Run:
        """
        Run the task-specific evaluation function

        Returns:
            Run: The evaluation run with score, steps, and task_id
        """
        url = f"{settings.base_url}/environments/{self.env_id}/evaluate"
        response = await make_request(url, "PUT", settings.api_key)
        run = Run(**response)
        # Note: Environment stores session name as string, not Session object
        # Session runs should be managed at the Session level, not here
        return run
    
    async def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment, optionally to a different task

        Args:
            task_id: Optional new task ID to reset to

        Returns:
            Observation: The initial observation of the environment
        """
        reset_task_id = task_id or self.task_id
        url = f"{settings.base_url}/environments/{self.env_id}/reset?task_id={reset_task_id}"
        response = await make_request(url, "PUT", settings.api_key)
        self.current_obs = Observation(**response["current_obs"])
        self.task_id = response["task_id"]
        return self.current_obs

    async def close(self) -> None:
        """
        Close the environment
        """
        url = f"{settings.base_url}/environments/{self.env_id}/close"
        await make_request(url, "PUT", settings.api_key)
        self.status = EnvStatus.CLOSED


