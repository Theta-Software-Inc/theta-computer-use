# openai_cua.py
from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI

from theta.models import (
    Observation,
    Action,
    ClickAction,
    ScrollAction,
    MoveAction,
    DragAction,
    TypeAction,
    KeyPressAction,
    Point,
    Key,
)


class OpenAIAgent:
    def __init__(
        self,
        name: str,
        api_key: str,
        model: str = "computer-use-preview",
        screen_size: tuple[int, int] = (1024, 768),
    ):
        self.name = name
        self.openai_client = OpenAI(api_key=api_key)
        self.model = model
        self.width, self.height = screen_size

        self._prev_response_id: Optional[str] = None
        self._pending_call_id: Optional[str] = None
        self._pending_safety: List[Dict[str, Any]] = []
        self._sx: float = 1.0
        self._sy: float = 1.0
        
        # Trajectory tracking
        self._trajectory: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._session_start_time = datetime.now().isoformat()

        self._system_prompt = (
            "You are a computer-use agent. Do not ask questions or wait for clarification. "
            "Use the computer tool autonomously until the task is complete. "
            "If you finish or cannot proceed further, end your response with 'done' or 'fail' without issuing a computer_call."
        )

        self._tools = [{
            "type": "computer_use_preview",
            "display_width": self.width,
            "display_height": self.height,
            "environment": "linux",
        }]

        self._reasoning = {"summary": "auto"}

    def act(self, obs: Observation) -> tuple[Optional[Action], bool]:
        """
        Return (Action, done). done=True if the model returns no computer_call.
        """
        self._step_count += 1
        step_start_time = datetime.now().isoformat()
        scaled = self.rescale(obs)
        
        # Initialize step data for trajectory
        step_data = {
            "step": self._step_count,
            "timestamp": step_start_time,
            "observation": {
                "text": obs.text,
                "screenshot": obs.screenshot,  # This will be the original base64
                "screenshot_size": None
            },
            "reasoning": None,
            "tool_call": None,
            "action_taken": None,
            "done": False
        }
        
        # Capture screenshot size if available
        if obs.screenshot:
            try:
                b64 = obs.screenshot.split(",", 1)[-1] if obs.screenshot.startswith("data:") else obs.screenshot
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                step_data["observation"]["screenshot_size"] = list(img.size)
            except:
                pass

        for attempt in range(12):
            inputs: List[Dict[str, Any]] = []

            if self._prev_response_id is None:
                user_content = obs.text or "Complete the task on screen."
                inputs.append({"type": "message", "role": "user", "content": user_content})
            elif self._pending_call_id:
                if not scaled.screenshot:
                    return None, True
                item: Dict[str, Any] = {
                    "type": "computer_call_output",
                    "call_id": self._pending_call_id,
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{scaled.screenshot}",
                    },
                }
                if self._pending_safety:
                    item["acknowledged_safety_checks"] = self._pending_safety
                inputs.append(item)
                self._pending_call_id = None
                self._pending_safety = []

            resp = self.openai_client.responses.create(
                model=self.model,
                input=inputs,
                tools=self._tools,
                previous_response_id=self._prev_response_id,
                reasoning=self._reasoning,
                truncation="auto",
                instructions=self._system_prompt,
            )
            self._prev_response_id = getattr(resp, "id", None)
            
            # Get output first so we can check for reasoning there too
            output = getattr(resp, "output", None) or []
            
            # Capture reasoning from the response - extract actual reasoning text
            reasoning_text = None
            if hasattr(resp, 'reasoning') and resp.reasoning:
                # Try to get the actual reasoning content/text, not just config
                if hasattr(resp.reasoning, 'content'):
                    reasoning_text = resp.reasoning.content
                elif hasattr(resp.reasoning, 'text'):
                    reasoning_text = resp.reasoning.text
                elif isinstance(resp.reasoning, str):
                    reasoning_text = resp.reasoning
                else:
                    # Fallback: check if reasoning is in the response output
                    reasoning_text = None
            
            # Also check for reasoning in the response output messages
            if not reasoning_text and output:
                for item in output:
                    if getattr(item, "type", None) == "reasoning" and hasattr(item, 'content'):
                        reasoning_text = item.content
                        break
                    elif getattr(item, "type", None) == "message" and hasattr(item, 'reasoning'):
                        reasoning_text = getattr(item.reasoning, 'content', None) or getattr(item.reasoning, 'text', None)
                        break
            
            step_data["reasoning"] = reasoning_text if reasoning_text else "No reasoning available"
            calls = [it for it in output if getattr(it, "type", None) == "computer_call"]

            if not calls:
                step_data["done"] = True
                step_data["tool_call"] = None
                step_data["action_taken"] = None
                self._trajectory.append(step_data)
                return None, True

            call = calls[0]
            self._pending_call_id = getattr(call, "call_id", None)
            self._pending_safety = getattr(call, "pending_safety_checks", []) or []
            action = getattr(call, "action", {}) or {}
            a_type = str(getattr(action, "type", "")).lower()
            
            # Capture tool call information
            step_data["tool_call"] = {
                "call_id": self._pending_call_id,
                "action_type": a_type,
                "action_data": self._serialize_object(action),
                "attempt": attempt + 1
            }

            if a_type in ("screenshot", "wait"):
                continue

            ua = self.translate(action)
            
            # Capture the final action taken
            step_data["action_taken"] = {
                "type": ua.type if hasattr(ua, 'type') else type(ua).__name__,
                "parameters": self._serialize_object(ua)
            }
            step_data["done"] = False
            
            self._trajectory.append(step_data)
            return ua, False

        # If we exhausted all attempts without taking action
        step_data["done"] = True
        step_data["tool_call"] = None
        step_data["action_taken"] = None
        self._trajectory.append(step_data)
        return None, True

    def rescale(self, obs: Observation) -> Observation:
        if not obs or not obs.screenshot:
            return obs

        b64 = obs.screenshot.split(",", 1)[-1] if obs.screenshot.startswith("data:") else obs.screenshot
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        ow, oh = img.size

        self._sx = (ow / float(self.width)) if self.width else 1.0
        self._sy = (oh / float(self.height)) if self.height else 1.0

        if (ow, oh) != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return Observation(screenshot=out_b64, text=obs.text)

    def translate(self, action: Any) -> Action:
        t = str(getattr(action, "type", "")).lower()

        def v(name: str, default=None):
            return getattr(action, name, default)

        def map_xy(x: float, y: float) -> Tuple[float, float]:
            return float(x) * self._sx, float(y) * self._sy

        if t == "click":
            x, y = map_xy(v("x", 0), v("y", 0))
            return ClickAction(type="click", point=Point(x=x, y=y), button=v("button", "left"))
        if t == "double_click":
            x, y = map_xy(v("x", 0), v("y", 0))
            return ClickAction(type="double_click", point=Point(x=x, y=y), button=v("button", "left"))
        if t == "move":
            x, y = map_xy(v("x", 0), v("y", 0))
            return MoveAction(type="move", point=Point(x=x, y=y))
        if t == "scroll":
            x, y = map_xy(v("x", 0), v("y", 0))
            dx, dy = map_xy(v("scroll_x", 0), v("scroll_y", 0))
            return ScrollAction(type="scroll", point=Point(x=x, y=y), scroll_delta=Point(x=dx, y=dy))
        if t == "drag":
            pts: List[Point] = []
            for p in (v("path", []) or []):
                # Handle different path point formats
                if isinstance(p, dict):
                    px, py = p.get("x", 0), p.get("y", 0)
                elif hasattr(p, 'x') and hasattr(p, 'y'):
                    # Handle objects with x,y attributes (like ActionDragPath)
                    px, py = getattr(p, 'x', 0), getattr(p, 'y', 0)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    # Handle list/tuple format
                    px, py = p[0], p[1]
                else:
                    # Fallback - try to extract any way we can
                    px = getattr(p, 'x', 0) if hasattr(p, 'x') else 0
                    py = getattr(p, 'y', 0) if hasattr(p, 'y') else 0
                
                mx, my = map_xy(px, py)
                pts.append(Point(x=mx, y=my))
            return DragAction(type="drag", path=pts)
        if t == "type":
            return TypeAction(type="type_text", text=str(v("text", "")))
        if t == "keypress":
            keys = [Key(key=str(k)) for k in (v("keys", []) or [])]
            return KeyPressAction(type="key_press", keys=keys)

        raise NotImplementedError(f"Unhandled action type: {t}")

    def get_response(self, prompt: str) -> str:
        r = self.openai_client.responses.create(
            model=self.model,
            input=[{"type": "message", "role": "user", "content": prompt}],
            tools=self._tools,
            reasoning=self._reasoning,
            truncation="auto",
            instructions=self._system_prompt,
        )
        for item in getattr(r, "output", []) or []:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", []) or []
                if content and isinstance(content, list):
                    seg = content[0]
                    if isinstance(seg, dict) and seg.get("type") in ("output_text", "text"):
                        return seg.get("text", "")
        return ""
    
    def _serialize_object(self, obj: Any) -> Any:
        """
        Safely serialize an object for JSON storage.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_object(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_object(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return {k: self._serialize_object(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        else:
            return str(obj)
    
    def trajectory_json(self, output_dir: str = "trajectories", save_images: bool = True, eval_score: Optional[float] = None) -> str:
        """
        Export the trajectory to a JSON file with optional image saving.
        
        Args:
            output_dir: Directory to save the trajectory file (default: "trajectories")
            save_images: Whether to save screenshots as separate PNG files (default: True)
            eval_score: Optional final evaluation score to include in trajectory metadata
            
        Returns:
            str: Path to the saved trajectory file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create unique trajectory session folder
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"{self.name}_trajectory_{session_timestamp}")
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        # Create images subdirectory
        images_dir = os.path.join(session_dir, "images")
        if save_images and not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Create filename
        filename = f"{self.name}_trajectory.json"
        filepath = os.path.join(session_dir, filename)
        
        # Process trajectory data, saving images separately
        processed_trajectory = []
        for step in self._trajectory:
            processed_step = step.copy()
            
            if save_images and step.get("observation", {}).get("screenshot"):
                screenshot_b64 = step["observation"]["screenshot"]
                
                # Save screenshot as PNG file
                step_num = str(step["step"]).zfill(3)
                image_filename = f"step_{step_num}.png"
                image_path = os.path.join(images_dir, image_filename)
                
                try:
                    # Decode and save the image
                    if screenshot_b64.startswith("data:"):
                        screenshot_b64 = screenshot_b64.split(",", 1)[1]
                    
                    image_data = base64.b64decode(screenshot_b64)
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_data)
                    
                    # Store relative path instead of base64
                    processed_step["observation"]["screenshot"] = f"images/{image_filename}"
                    processed_step["observation"]["screenshot_saved"] = True
                except Exception as e:
                    processed_step["observation"]["screenshot"] = f"Error saving image: {str(e)}"
                    processed_step["observation"]["screenshot_saved"] = False
            else:
                # If not saving images, remove screenshot data to reduce size
                if processed_step.get("observation"):
                    processed_step["observation"]["screenshot"] = None
                    processed_step["observation"]["screenshot_saved"] = False
            
            processed_trajectory.append(processed_step)
        
        # Prepare trajectory data
        trajectory_data = {
            "agent_name": self.name,
            "model": self.model,
            "screen_size": [self.width, self.height],
            "session_start_time": self._session_start_time,
            "total_steps": len(processed_trajectory),
            "eval_score": eval_score,
            "images_saved": save_images,
            "images_directory": "images/" if save_images else None,
            "session_directory": session_dir,
            "trajectory": processed_trajectory
        }
        
        # Write to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Trajectory saved to: {filepath}")
        if save_images:
            image_count = sum(1 for step in self._trajectory if step.get("observation", {}).get("screenshot"))
            print(f"🖼️  {image_count} screenshots saved to: {images_dir}/")
        
        # Store session directory for HTML viewer
        self._current_session_dir = session_dir
        
        return filepath
    
    def trajectory_html_viewer(self, output_dir: str = "trajectories", eval_score: Optional[float] = None) -> str:
        """
        Generate an HTML viewer for the trajectory with images.
        
        Args:
            output_dir: Directory containing the trajectory files (unused if session dir exists)
            eval_score: Optional evaluation score to display in the viewer header
            
        Returns:
            str: Path to the generated HTML file
        """
        # Use the session directory if available, otherwise create a new one
        if hasattr(self, '_current_session_dir') and self._current_session_dir:
            session_dir = self._current_session_dir
        else:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create unique trajectory session folder
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(output_dir, f"{self.name}_trajectory_{session_timestamp}")
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)
        
        html_filename = f"{self.name}_trajectory_viewer.html"
        html_path = os.path.join(session_dir, html_filename)
        
        # HTML template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory Viewer - {self.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step {{ background: white; margin-bottom: 15px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .step-header {{ background: #007acc; color: white; padding: 15px; font-weight: bold; }}
        .step-content {{ padding: 20px; }}
        .screenshot {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
        .action-info {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .reasoning {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .metadata {{ font-size: 0.9em; color: #666; }}
        .json-code {{ background: #f4f4f4; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 0.9em; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agent Trajectory: {self.name}</h1>
            <p><strong>Model:</strong> {self.model}</p>
            <p><strong>Screen Size:</strong> {self.width} x {self.height}</p>
            <p><strong>Session Start:</strong> {self._session_start_time}</p>
            <p><strong>Total Steps:</strong> {len(self._trajectory)}</p>
            {f'<p><strong>Evaluation Score:</strong> {eval_score:.3f}</p>' if eval_score is not None else ''}
        </div>
"""

        for step in self._trajectory:
            step_num = step.get('step', 0)
            timestamp = step.get('timestamp', 'Unknown')
            observation = step.get('observation', {})
            reasoning = step.get('reasoning', '')
            tool_call = step.get('tool_call', {})
            action_taken = step.get('action_taken', {})
            done = step.get('done', False)
            
            html_content += f"""
        <div class="step">
            <div class="step-header">
                Step {step_num} - {timestamp}
                {'✅ Done' if done else '⚡ In Progress'}
            </div>
            <div class="step-content">
"""
            
            # Add screenshot if available
            screenshot_path = observation.get('screenshot')
            if screenshot_path and not screenshot_path.startswith('Error'):
                html_content += f"""
                <div>
                    <h3>📸 Screenshot</h3>
                    <img src="{screenshot_path}" alt="Step {step_num} screenshot" class="screenshot">
                </div>
"""
            
            # Add observation text
            if observation.get('text'):
                html_content += f"""
                <div>
                    <h3>👁️ Observation Text</h3>
                    <p>{observation['text']}</p>
                </div>
"""
            
            # Add reasoning if available
            if reasoning:
                reasoning_str = str(reasoning) if reasoning else "No reasoning available"
                html_content += f"""
                <div class="reasoning">
                    <h3>🧠 Model Reasoning</h3>
                    <p>{reasoning_str}</p>
                </div>
"""
            
            # Add tool call information
            if tool_call:
                html_content += f"""
                <div class="action-info">
                    <h3>🔧 Tool Call</h3>
                    <p><strong>Action Type:</strong> {tool_call.get('action_type', 'Unknown')}</p>
                    <p><strong>Attempt:</strong> {tool_call.get('attempt', 1)}</p>
                </div>
"""
            
            # Add action taken
            if action_taken:
                html_content += f"""
                <div class="action-info">
                    <h3>⚡ Action Taken</h3>
                    <p><strong>Type:</strong> {action_taken.get('type', 'Unknown')}</p>
                    <div class="json-code">{json.dumps(action_taken.get('parameters', {}), indent=2)}</div>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML viewer saved to: {html_path}")
        print(f"📱 Open in browser: file://{os.path.abspath(html_path)}")
        print(f"📁 Complete trajectory session saved in: {session_dir}")
        return html_path
    
    def clear_trajectory(self):
        """Clear the current trajectory data."""
        self._trajectory = []
        self._step_count = 0
        self._session_start_time = datetime.now().isoformat()
        if hasattr(self, '_current_session_dir'):
            delattr(self, '_current_session_dir')
        print("🧹 Trajectory cleared")
