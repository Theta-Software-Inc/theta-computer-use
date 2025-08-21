import httpx
from typing import Any, Optional, Dict

async def make_request(
    url: str, 
    method: str, 
    api_key: Optional[str] = None, 
    data: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0
) -> Any:
    """
    Make a request to the Theta Computer Use Environments API

    Args:
        url (str): The URL to make the request to
        method (str): The HTTP method to use
        api_key (Optional[str]): The API key to use
        data (Optional[Any]): The JSON data to send with the request
        params (Optional[Dict[str, Any]]): Query parameters to send with the request
        timeout (float): Request timeout in seconds (default: 200.0)

    Returns:
        Any: The response from the API (dict or raw response for string responses)

    Raises:
        ValueError: If the API key is missing
    """
    if not api_key:
        raise ValueError("API key is missing")
        
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method, 
            url, 
            json=data, 
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        # Some endpoints return plain strings (like VNC URL)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        else:
            # Return the text response for string endpoints
            return response.text.strip('"')  # Remove quotes if present
