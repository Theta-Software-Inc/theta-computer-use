from typing import Optional, List
from .models import EnvStatus, Observation
from .session import Session
from .environment import Environment
from .requests import make_request
from .settings import settings


class Client:
    """
    Client for the Theta Computer Use Environments
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the client

        Args:
            api_key (str): The API authentication key
        """
        if api_key:
            settings.api_key = api_key
        
    async def get_sessions(self) -> List[Session]:
        """
        Get all sessions for the authenticated user

        Returns:
            List[Session]: A list of Session objects
        """
        url = f"{settings.base_url}/sessions/"
        response = await make_request(url, "GET", settings.api_key)
        return [Session(name=session["name"], runs=session.get("runs")) for session in response]
    
    async def create_session(self, name: str) -> Session:
        """
        Create a new session

        Args:
            name (str): The name of the session

        Returns:
            Session: The created session object
        """
        url = f"{settings.base_url}/sessions/"
        response = await make_request(url, "POST", settings.api_key, {"name": name})
        return Session(name=response["name"], runs=response.get("runs"))
    
    async def get_tasks(self) -> List[str]:
        """
        Get all task IDs the user is authorized to access

        Returns:
            List[str]: A list of task IDs
        """
        url = f"{settings.base_url}/tasks"
        response = await make_request(url, "GET", settings.api_key)
        return response
    
    async def list_environments(self) -> List[Environment]:
        """
        List all running environments for the authenticated user

        Returns:
            List[Environment]: A list of running Environment objects
        """
        url = f"{settings.base_url}/environments"
        response = await make_request(url, "GET", settings.api_key)
        return [
            Environment(
                env_id=env["env_id"],
                session=env["session"],
                task_id=env["task_id"],
                status=EnvStatus(env["status"]),
                current_obs=Observation(**env["current_obs"]) if env.get("current_obs") else None,
                vnc_url=env.get("vnc_url")
            )
            for env in response
        ]
    
    async def close_all_environments(self) -> dict:
        """
        Close all running environments for the authenticated user

        Returns:
            dict: Summary with counts of closed, failed, and skipped environments
        """
        url = f"{settings.base_url}/environments/close_all"
        response = await make_request(url, "PUT", settings.api_key, timeout=200.0)
        return response