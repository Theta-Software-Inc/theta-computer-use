from typing import List, Optional
from .environment import Environment
from .models import Run, EnvStatus, Observation, CreateEnvRequest
from .requests import make_request
from .settings import settings
import logging

class Session:
    """
    Session class for the Theta Computer Use Environments
    """
    def __init__(self, name: str, runs: Optional[List[Run]] = None) -> None:
        """
        Initialize a session

        Args:
            name (str): The name of the session
            runs (Optional[List[Run]]): List of evaluation runs
        """
        self.name = name
        self.envs = []
        self.runs = runs or []

    async def create_environment(self, task_id: str) -> Environment:
        """Create a new environment.

        Args:
            task_id: The task ID to run in the environment

        Returns:
            Environment: The created environment
        """
        created_envs = await self.create_environments([task_id])
        if created_envs:
            return created_envs[0]
        logging.error(f"Failed to create environment for task {task_id}")
        return None
    
    async def create_environments(self, task_ids: List[str]) -> List[Environment]:
        """Create multiple environments in parallel.

        Args:
            task_ids: A single task ID or list of task IDs to run in the environments

        Returns:
            List[Environment]: List of successfully created environments
        """
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        
        url = f"{settings.base_url}/environments/"
        request = CreateEnvRequest(session=self.name, task_ids=task_ids)
        response = await make_request(
            url, 
            "POST", 
            api_key=settings.api_key, 
            data=request.model_dump()
        )
        
        # Create Environment objects directly from API response
        created_environments = []
        for env_data in response["environments"]:
            environment = Environment(
                env_id=env_data["env_id"],
                session=env_data["session"],
                task_id=env_data["task_id"],
                status=EnvStatus(env_data["status"]),
                current_obs=Observation(**env_data["current_obs"]) if env_data.get("current_obs") else None
            )
            self.envs.append(environment)
            created_environments.append(environment)
        logging.info(f"{len(created_environments)} environments requested, {response['created_count']} created, {response['failed_count']} failed")
        
        return created_environments