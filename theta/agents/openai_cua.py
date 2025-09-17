# openai_cua.py
from __future__ import annotations

import base64
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import AsyncOpenAI

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
    """
    OpenAI Computer Use (Responses API) agent.

    Key points:
    - Uses the Responses API with the `computer_use_preview` tool.
    - Handles `screenshot` and `wait` inline by replying with `computer_call_output` that
      contains a fresh `input_image` data URL.
    - Translates only the documented OpenAI action set:
        click, double_click, move, scroll, drag, type, keypress
      (screenshot/wait are handled inline and not translated to universal actions).
    - Scroll deltas (scroll_x / scroll_y) are treated as wheel units (not pixels):
        vertical: + = up, - = down
        horizontal: + = right, - = left
      -> We DO NOT scale the amount, but we DO scale the target coordinate x,y.
    - Captures all available reasoning items and stores them in the trajectory.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str = "computer-use-preview",
        screen_size: tuple[int, int] = (1024, 768),
        *,
        enable_logging: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.name = name
        self.openai_client = AsyncOpenAI(api_key=api_key)
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
        self._enable_logging = enable_logging
        self.logger = logger or logging.getLogger("theta.agents.openai")
        if self._enable_logging:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                _h = logging.StreamHandler()
                _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self.logger.addHandler(_h)
            self.logger.propagate = False

        self._system_prompt = (
            "You are a computer-use agent. Do not ask questions or wait for clarification. "
            "Use the computer tool autonomously until the task is complete. "
            "If you finish or cannot proceed further, end your response with 'done' or 'fail' without issuing another computer call. "
            "If the request includes any files (e.g. PDFs), treat them as provided materials and consult them directly to complete the task instead of asking the user for attachments or summaries."
        )

        self._tools = [{
            "type": "computer_use_preview",
            "display_width": self.width,
            "display_height": self.height,
            # environment: "linux"|"windows"|"mac" – either is fine for display-only flows
            "environment": "linux",
        }]

        # Ask the API to include a concise reasoning summary where available
        self._reasoning = {"summary": "auto"}

        # Materials (PDFs) to attach on the first request, as Responses 'input_file' parts
        self._materials_segments: List[Dict[str, Any]] = []

    # --------------------------- public API ---------------------------

    def set_materials(self, materials: Dict[str, str]) -> None:
        """
        Accepts dict[str, str]: {filename: raw_base64_pdf}
        Converts to Responses 'input_file' parts (data URL) and stores them
        to be attached on the *first* request only.
        (No base64 validation / extra logging; we normalize data URLs if present.)
        """
        self._materials_segments = []
        if not isinstance(materials, dict):
            return
        for filename, raw in materials.items():
            s = (raw or "").strip()
            if not s:
                continue
            # If a data URL slipped in, reduce to raw base64 to avoid double prefixing
            if s.startswith("data:") and "base64," in s:
                s = s.split("base64,", 1)[1]
            self._materials_segments.append({
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{s}",
            })

    async def act(self, obs: Observation) -> tuple[Optional[Action], bool]:
        """
        Step the agent once with the given observation.
        Returns (Action, done).
        - If 'done' is True, there is no Action to execute (model ended or waiting).
        - If 'done' is False, execute the Action and then call act() again with the new obs.
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
                "screenshot": obs.screenshot,   # original base64 you provided
                "screenshot_size": None
            },
            "reasoning": None,
            "reasoning_blocks": [],  # raw blocks
            "assistant_text": None,
            "tool_call": None,
            "tool_call_raw": None,   # full raw call object
            "action_taken": None,
            "done": False
        }

        # Capture original screenshot size
        if obs.screenshot:
            try:
                b64 = obs.screenshot.split(",", 1)[-1] if obs.screenshot.startswith("data:") else obs.screenshot
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                step_data["observation"]["screenshot_size"] = list(img.size)
            except Exception:
                pass

        for attempt in range(12):
            inputs: List[Dict[str, Any]] = []

            # First turn: send instruction + (optional) PDFs + (optional) initial screenshot
            if self._prev_response_id is None:
                has_materials = bool(self._materials_segments)
                default_text = (
                    "Use the attached PDF(s) and the on-screen image to complete the task instructions."
                    if has_materials else
                    "Use the on-screen image to complete the task instructions."
                )
                user_content = obs.text or default_text

                # Build a structured content message
                parts: List[Dict[str, Any]] = []
                if has_materials:
                    parts.extend(self._materials_segments)
                if scaled.screenshot:
                    parts.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{scaled.screenshot}",
                    })
                parts.append({"type": "input_text", "text": user_content})

                inputs.append({
                    "type": "message",
                    "role": "user",
                    "content": parts,
                })

            # If we owe a screenshot from a prior computer_call, send it now
            elif self._pending_call_id:
                if not scaled.screenshot:
                    # no image available => end gracefully
                    step_data["done"] = True
                    self._trajectory.append(step_data)
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

            # Ask the model to proceed
            resp = await self.openai_client.responses.create(
                model=self.model,
                input=inputs,
                tools=self._tools,
                previous_response_id=self._prev_response_id,
                reasoning=self._reasoning,
                truncation="auto",
                instructions=self._system_prompt,
            )
            self._prev_response_id = getattr(resp, "id", None)

            # ------ capture reasoning / thinking ------
            output = getattr(resp, "output", None) or []

            # Raw reasoning blocks (from output items)
            rblocks: List[Any] = []
            try:
                for it in output:
                    if getattr(it, "type", None) == "reasoning":
                        rblocks.append(self._serialize_object(it))
                    elif getattr(it, "type", None) == "message" and hasattr(it, "reasoning") and it.reasoning:
                        rblocks.append(self._serialize_object(it.reasoning))
            except Exception:
                pass

            # Human-readable reasoning summary if present
            reasoning_text = None
            if hasattr(resp, 'reasoning') and resp.reasoning:
                try:
                    summ = getattr(resp.reasoning, "summary", None)
                    if isinstance(summ, list) and summ:
                        maybe = getattr(summ[0], "text", None) or getattr(summ[0], "content", None)
                        if maybe:
                            reasoning_text = str(maybe)
                except Exception:
                    pass

            if not reasoning_text and rblocks:
                for rb in rblocks:
                    if isinstance(rb, dict):
                        text = rb.get("text") or rb.get("content")
                        if text:
                            reasoning_text = str(text)
                            break

            step_data["reasoning_blocks"] = rblocks
            step_data["reasoning"] = reasoning_text or "No reasoning available"

            # Assistant text outputs (human-readable)
            texts: List[str] = []
            try:
                for it in output:
                    if getattr(it, "type", None) == "message":
                        content = getattr(it, "content", []) or []
                        if isinstance(content, list):
                            for seg in content:
                                if isinstance(seg, dict) and seg.get("type") in ("output_text", "text"):
                                    txt = seg.get("text")
                                    if txt:
                                        texts.append(str(txt))
            except Exception:
                pass
            step_data["assistant_text"] = "\n".join(texts) if texts else None

            # ------ find and handle computer calls ------
            calls = [it for it in output if getattr(it, "type", None) == "computer_call"]
            if not calls:
                # model ended or produced only text
                step_data["done"] = True
                self._trajectory.append(step_data)
                self._log_step(step_data)
                return None, True

            call = calls[0]
            self._pending_call_id = getattr(call, "call_id", None)
            self._pending_safety = getattr(call, "pending_safety_checks", []) or []
            action = getattr(call, "action", {}) or {}
            a_type = str(getattr(action, "type", "")).lower()

            # record exact tool call for debugging/auditing
            step_data["tool_call"] = {
                "call_id": self._pending_call_id,
                "action_type": a_type,
                "action_data": self._serialize_object(action),
                "attempt": attempt + 1
            }
            step_data["tool_call_raw"] = self._serialize_object(call)

            # 'screenshot' and 'wait' are handled by sending a fresh image next loop
            if a_type in ("screenshot", "wait"):
                continue

            # translate to universal Action
            ua = self.translate(action)

            step_data["action_taken"] = {
                "type": getattr(ua, 'type', type(ua).__name__),
                "parameters": self._serialize_object(ua)
            }
            step_data["done"] = False
            self._trajectory.append(step_data)
            self._log_step(step_data)
            return ua, False

        # Exhausted attempts without a concrete action
        step_data["done"] = True
        self._trajectory.append(step_data)
        self._log_step(step_data)
        return None, True

    def trajectory_json(self, output_dir: str = "trajectories", save_images: bool = True, eval_score: Optional[float] = None) -> str:
        """
        Save the full trajectory to disk; optionally export screenshots to files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uniq = uuid.uuid4().hex[:6]
        session_dir = os.path.join(output_dir, f"{self.name}_trajectory_{session_timestamp}-{uniq}")
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

        images_dir = os.path.join(session_dir, "images")
        if save_images and not os.path.exists(images_dir):
            os.makedirs(images_dir)

        filename = f"{self.name}_trajectory.json"
        filepath = os.path.join(session_dir, filename)

        processed_trajectory = []
        for step in self._trajectory:
            processed_step = step.copy()
            # Standardized labels to match logging
            processed_step["text_output"] = processed_step.get("assistant_text")
            processed_step["computer_tool_call"] = processed_step.get("tool_call_raw")
            processed_step["translated_action"] = processed_step.get("action_taken")
            if save_images and step.get("observation", {}).get("screenshot"):
                screenshot_b64 = step["observation"]["screenshot"]
                step_num = str(step["step"]).zfill(3)
                image_filename = f"step_{step_num}.png"
                image_path = os.path.join(images_dir, image_filename)
                try:
                    if screenshot_b64.startswith("data:"):
                        screenshot_b64 = screenshot_b64.split(",", 1)[1]
                    image_data = base64.b64decode(screenshot_b64)
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_data)
                    processed_step["observation"]["screenshot"] = f"images/{image_filename}"
                    processed_step["observation"]["screenshot_saved"] = True
                except Exception as e:
                    processed_step["observation"]["screenshot"] = f"Error saving image: {str(e)}"
                    processed_step["observation"]["screenshot_saved"] = False
            else:
                if processed_step.get("observation"):
                    processed_step["observation"]["screenshot"] = None
                    processed_step["observation"]["screenshot_saved"] = False
            processed_trajectory.append(processed_step)

        data = {
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

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        steps_count = len(processed_trajectory)
        print(f"✅ {steps_count} trajectory steps saved in: {session_dir}")
        self._current_session_dir = session_dir
        return filepath

    def trajectory_html_viewer(self, output_dir: str = "trajectories", eval_score: Optional[float] = None) -> str:
        """
        Generate a simple HTML viewer (kept identical across agents for parity).
        """
        if hasattr(self, '_current_session_dir') and self._current_session_dir:
            session_dir = self._current_session_dir
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(output_dir, f"{self.name}_trajectory_{session_timestamp}")
            if not os.path.exists(session_dir):
                os.makedirs(session_dir)

        html_filename = f"{self.name}_trajectory_viewer.html"
        html_path = os.path.join(session_dir, html_filename)

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

            screenshot_path = observation.get('screenshot')
            if screenshot_path and not screenshot_path.startswith('Error'):
                html_content += f"""
      <div>
        <h3>📸 Screenshot</h3>
        <img src="{screenshot_path}" alt="Step {step_num} screenshot" class="screenshot">
      </div>
"""

            # Observation text (initial task instructions or subsequent text observation)
            obs_text = observation.get('text') or "No text observation available"
            html_content += f"""
      <div>
        <h3>Observation Text:</h3>
        <p>{obs_text}</p>
      </div>
"""

            # Text Output (assistant text) – only render if present
            if step.get('assistant_text'):
                html_content += f"""
      <div>
        <h3>Text Output:</h3>
        <p>{step.get('assistant_text')}</p>
      </div>
"""

            if reasoning:
                reasoning_str = str(reasoning) if reasoning else "No reasoning available"
                html_content += f"""
      <div class="reasoning">
        <h3>Reasoning:</h3>
        <p>{reasoning_str}</p>
      </div>
"""

            if tool_call:
                html_content += f"""
      <div class="action-info">
        <h3>Computer Tool Call:</h3>
        <div class="json-code">{json.dumps(step.get('tool_call_raw'), indent=2)}</div>
      </div>
"""

            if action_taken:
                html_content += f"""
      <div class="action-info">
        <h3>Translated Action:</h3>
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

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📱 Open in browser: file://{os.path.abspath(html_path)}")
        return html_path

    def clear_trajectory(self):
        """Clear current trajectory data."""
        self._trajectory = []
        self._step_count = 0
        self._session_start_time = datetime.now().isoformat()
        if hasattr(self, '_current_session_dir'):
            delattr(self, '_current_session_dir')
        print("🧹 Trajectory cleared")

    # --------------------------- helpers ---------------------------

    def rescale(self, obs: Observation) -> Observation:
        """
        Resize the screenshot to the tool display size for the model,
        and set sx/sy so model coordinates can be mapped back.
        """
        if not obs or not obs.screenshot:
            return obs

        b64 = obs.screenshot.split(",", 1)[-1] if obs.screenshot.startswith("data:") else obs.screenshot
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        ow, oh = img.size

        # mapping from tool display coords -> original pixels
        self._sx = (ow / float(self.width)) if self.width else 1.0
        self._sy = (oh / float(self.height)) if self.height else 1.0

        if (ow, oh) != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return Observation(screenshot=out_b64, text=obs.text)

    def translate(self, action: Any) -> Action:
        """
        Translate OpenAI CUA computer_call.action into a theta Action.
        Only the documented OpenAI action space is supported.
        """
        t = str(getattr(action, "type", "")).lower()

        def v(name: str, default=None):
            return getattr(action, name, default)

        def map_xy(x: float, y: float) -> Tuple[float, float]:
            # scale from tool display space -> original pixels
            return float(x) * self._sx, float(y) * self._sy

        if t in ("click", "left_click"):
            x, y = map_xy(v("x", 0), v("y", 0))
            return ClickAction(type="click", point=Point(x=x, y=y), button=v("button", "left"))

        if t == "double_click":
            x, y = map_xy(v("x", 0), v("y", 0))
            return ClickAction(type="double_click", point=Point(x=x, y=y), button=v("button", "left"))

        if t in ("move", "mouse_move"):
            x, y = map_xy(v("x", 0), v("y", 0))
            return MoveAction(type="move", point=Point(x=x, y=y))

        if t == "scroll":
            # Treat scroll_x/scroll_y as wheel units (no scaling of amount).
            # Scale only the target coordinate. Engine expects down as negative.
            x, y = map_xy(v("x", 0), v("y", 0))
            dx = int(v("scroll_x", 0) or 0)
            dy_raw = int(v("scroll_y", 0) or 0)
            dy = -dy_raw
            return ScrollAction(type="scroll", point=Point(x=x, y=y), scroll_delta=Point(x=dx, y=dy))

        if t == "drag":
            pts: List[Point] = []
            for p in (v("path", []) or []):
                if isinstance(p, dict):
                    px, py = p.get("x", 0), p.get("y", 0)
                elif hasattr(p, 'x') and hasattr(p, 'y'):
                    px, py = getattr(p, 'x', 0), getattr(p, 'y', 0)
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    px, py = p[0], p[1]
                else:
                    px = getattr(p, 'x', 0) if hasattr(p, 'x') else 0
                    py = getattr(p, 'y', 0) if hasattr(p, 'y') else 0
                mx, my = map_xy(px, py)
                pts.append(Point(x=mx, y=my))
            if not pts:
                raise NotImplementedError("drag requires at least one point")
            return DragAction(type="drag", path=pts)

        if t == "type":
            return TypeAction(type="type_text", text=str(v("text", "")))

        if t == "keypress":
            keys = [Key(key=str(k)) for k in (v("keys", []) or [])]
            return KeyPressAction(type="key_press", keys=keys)

        # OpenAI CUA does not include triple_click / left_mouse_* / hold_key in its set
        if t in ("triple_click", "left_mouse_down", "left_mouse_up", "hold_key"):
            raise NotImplementedError(f"Unsupported OpenAI action: {t}")

        if t in ("screenshot", "wait"):
            # handled in act() by sending a screenshot; never translated here
            raise NotImplementedError(f"Internal-only action surfaced to translate(): {t}")

        raise NotImplementedError(f"Unhandled action type: {t}")

    async def get_response(self, prompt: str) -> str:
        """
        Convenience method for a one-off text reply (rarely used in CUA runs).
        """
        r = await self.openai_client.responses.create(
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

    # --------------------------- utility ---------------------------

    def _serialize_object(self, obj: Any) -> Any:
        """
        Safely serialize unknown SDK objects for JSON trajectory storage.
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

    def _log_step(self, step_data: Dict[str, Any]) -> None:
        if not getattr(self, "_enable_logging", False):
            return
        try:
            # Build a sanitized event for logging (exclude raw screenshots)
            step = step_data.get("step")
            
            reasoning = step_data.get("reasoning")
            assistant_text = step_data.get("assistant_text")
            tool_call_raw = step_data.get("tool_call_raw")
            action_taken = step_data.get("action_taken")

            lines: List[str] = []
            # Header is provided by logger formatter; include step number in message header line
            lines.append(f"Step {step}")
            lines.append("------------------------")
            lines.append("Reasoning:")
            lines.append(reasoning or "None")
            # For OpenAI CUA: include Text Output section only if we actually have assistant text
            if assistant_text:
                lines.append("------------------------")
                lines.append("Text Output:")
                lines.append(assistant_text)
            lines.append("------------------------")
            lines.append("Computer Tool Call:")
            lines.append(json.dumps(tool_call_raw, indent=2, ensure_ascii=False) if tool_call_raw is not None else "None")
            lines.append("------------------------")
            lines.append("Translated Action:")
            lines.append(json.dumps(action_taken, indent=2, ensure_ascii=False) if action_taken is not None else "None")
            lines.append("=====================")

            self.logger.info("\n".join(lines))
        except Exception:
            # Never allow logging to break control flow
            pass
