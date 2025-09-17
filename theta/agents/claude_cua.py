# claude_cua.py
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
from anthropic import AsyncAnthropic

from theta.models import (
    Observation,
    Action,
    ClickAction,
    ScrollAction,
    MoveAction,
    DragAction,
    TypeAction,
    KeyPressAction,
    MouseDownAction,
    MouseUpAction,
    HoldKeyAction,
    Point,
    Key,
)


class ClaudeAgent:
    """
    Anthropic Claude Computer Use agent (computer_20250124 by default).

    Key points:
    - Uses Messages API with the `computer_2025-01-24` (or 2024-10-22) tool.
    - Handles `screenshot` and `wait` inline by replying with a `tool_result`
      that contains a fresh image block (base64).
    - Translates the documented Claude action space:
        left_click, right_click, middle_click, double_click, triple_click,
        mouse_move, type, key, scroll (direction+amount),
        left_click_drag, left_mouse_down, left_mouse_up, hold_key
      (screenshot/wait handled inline).
    - For `scroll`: `scroll_amount` is a *number of wheel steps* (not pixels).
      -> We DO NOT scale the amount, but we DO scale the target coordinate x,y.
    - Captures all thinking / redacted_thinking blocks and stores them in the trajectory.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        screen_size: tuple[int, int] = (1024, 768),
        *,
        tool_version: str = "20250124",
        display_number: int = 1,
        max_tokens: int = 20000,
        enable_thinking: bool = True,
        enable_logging: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.name = name
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.width, self.height = screen_size
        self.max_tokens = max_tokens

        self._tool_type = f"computer_{tool_version}"
        self._betas = ["computer-use-2025-01-24"] if tool_version == "20250124" else ["computer-use-2024-10-22"]
        self._tools = [{
            "type": self._tool_type,
            "name": "computer",
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": display_number,
        }]
        self._thinking = {"type": "enabled", "budget_tokens": 10000} if enable_thinking else None

        self._system_prompt = (
            "You are a computer-use agent. Do not ask questions or wait for clarification. "
            "Use the computer tool autonomously until the task is complete. "
            "If you finish or cannot proceed further, end your response with 'done' or 'fail' without requesting more tools. "
            "If the request includes any 'document' blocks (e.g. PDFs), treat them as provided materials and consult them directly to complete the taskinstead of asking the user for attachments or summaries."
        )

        self._messages: List[Dict[str, Any]] = []
        self._pending_tool_use_id: Optional[str] = None
        self._pending_action_name: Optional[str] = None

        self._sx: float = 1.0
        self._sy: float = 1.0

        self._trajectory: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._session_start_time = datetime.now().isoformat()
        self._enable_logging = enable_logging
        self.logger = logger or logging.getLogger("theta.agents.claude")
        if self._enable_logging:
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                _h = logging.StreamHandler()
                _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self.logger.addHandler(_h)
            self.logger.propagate = False

        # Materials (PDFs) to attach on the first request as Messages 'document' blocks
        self._materials_blocks: List[Dict[str, Any]] = []
        self._materials_names: List[str] = []

    # --------------------------- public API ---------------------------

    def set_materials(self, materials: Dict[str, str]) -> None:
        """
        Accepts dict[str, str]: {filename: raw_base64_pdf}
        Converts to Anthropic 'document' blocks to be attached on the first request only.
        Assumes values are valid base64 (optionally prefixed with a data URL).
        """
        self._materials_blocks = []
        if not isinstance(materials, dict) or not materials:
            return

        for filename, raw in materials.items():
            s = str(raw or "").strip()
            if not s:
                continue
            # Normalize an optional data URL to raw base64
            if s.startswith("data:") and "base64," in s:
                s = s.split("base64,", 1)[1]

            self._materials_blocks.append({
                "type": "document",
                "title": filename or "material.pdf",
                "source": {"type": "base64", "media_type": "application/pdf", "data": s},
            })

    async def act(self, obs: Observation) -> tuple[Optional[Action], bool]:
        """
        Step the agent once with the given observation.
        Returns (Action, done).
        """
        self._step_count += 1
        step_start_time = datetime.now().isoformat()
        scaled = self.rescale(obs)

        step_data = {
            "step": self._step_count,
            "timestamp": step_start_time,
            "observation": {
                "text": obs.text,
                "screenshot": obs.screenshot,
                "screenshot_size": None,
            },
            "reasoning": None,
            "reasoning_blocks": [],  # raw thinking blocks
            "assistant_text": None,
            "tool_call": None,
            "tool_call_raw": None,
            "action_taken": None,
            "done": False,
        }

        # record original screenshot size
        if obs.screenshot:
            try:
                b64 = obs.screenshot.split(",", 1)[-1] if obs.screenshot.startswith("data:") else obs.screenshot
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                step_data["observation"]["screenshot_size"] = list(img.size)
            except Exception:
                pass

        for attempt in range(12):
            # If we owe a tool_result (screenshot), send it now
            if self._pending_tool_use_id:
                if not scaled.screenshot:
                    step_data["done"] = True
                    self._trajectory.append(step_data)
                    return None, True
                tool_result_block = self._make_tool_result_block(
                    self._pending_tool_use_id,
                    base64_png=scaled.screenshot,
                    note=f"Result after '{self._pending_action_name or 'action'}'",
                )
                self._append_user_message([tool_result_block])
                self._pending_tool_use_id = None
                self._pending_action_name = None

            # First turn: send the initial screenshot + instruction + materials
            if not self._messages:
                has_materials = bool(self._materials_blocks)
                default_text = (
                    "Use the attached PDF(s) and the on-screen image to complete the task instructions."
                    if has_materials else
                    "Use the on-screen image to complete the task instructions."
                )
                user_content = obs.text or default_text

                init_blocks: List[Dict[str, Any]] = []
                if has_materials:
                    init_blocks.extend(self._materials_blocks)
                if scaled.screenshot:
                    init_blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": scaled.screenshot}
                    })
                init_blocks.append({"type": "text", "text": user_content})
                self._append_user_message(init_blocks)

            params: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": self._messages,
                "tools": self._tools,
                "betas": self._betas,
                "system": self._system_prompt,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            }
            if self._thinking:
                params["thinking"] = self._thinking

            resp = await self.client.beta.messages.create(**params)
            self._append_assistant_message(resp.content)

            # ------ capture thinking / redacted_thinking ------
            thinking_blocks = self._collect_thinking_blocks(resp.content)
            step_data["reasoning_blocks"] = thinking_blocks
            step_data["reasoning"] = self._join_thinking_text(thinking_blocks) or "No reasoning available"

            # Assistant text blocks (human-readable)
            texts: List[str] = []
            try:
                for b in resp.content or []:
                    if self._safe_attr(b, "type") == "text":
                        t = self._safe_attr(b, "text")
                        if t:
                            texts.append(str(t))
            except Exception:
                pass
            step_data["assistant_text"] = "\n".join(texts) if texts else None

            # ------ find computer tool_use ------
            tool_uses = [b for b in (resp.content or []) if self._safe_attr(b, "type") == "tool_use" and self._safe_attr(b, "name") == "computer"]
            if not tool_uses:
                step_data["done"] = True
                self._trajectory.append(step_data)
                self._log_step(step_data)
                return None, True

            tu = tool_uses[0]
            tu_id = self._safe_attr(tu, "id")
            tu_input = self._safe_attr(tu, "input") or {}
            a_type = str(tu_input.get("action", "")).lower()

            step_data["tool_call"] = {
                "tool_use_id": tu_id,
                "action_type": a_type,
                "action_data": self._serialize_object(tu_input),
                "attempt": attempt + 1,
            }
            step_data["tool_call_raw"] = self._serialize_object(tu)

            # screenshot / wait → reply with a fresh image (tool_result)
            if a_type in ("screenshot", "wait"):
                if not scaled.screenshot:
                    step_data["done"] = True
                    self._trajectory.append(step_data)
                    self._log_step(step_data)
                    return None, True
                tool_result_block = self._make_tool_result_block(tu_id, base64_png=scaled.screenshot, note=a_type)
                self._append_user_message([tool_result_block])
                continue

            # translate to universal Action
            self._pending_tool_use_id = tu_id
            self._pending_action_name = a_type

            ua = self.translate(tu_input)

            step_data["action_taken"] = {
                "type": getattr(ua, "type", type(ua).__name__),
                "parameters": self._serialize_object(ua),
            }
            step_data["done"] = False
            self._trajectory.append(step_data)
            self._log_step(step_data)
            return ua, False

        step_data["done"] = True
        self._trajectory.append(step_data)
        self._log_step(step_data)
        return None, True


    def trajectory_json(self, output_dir: str = "trajectories", save_images: bool = True, eval_score: Optional[float] = None) -> str:
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
            "trajectory": processed_trajectory,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        steps_count = len(processed_trajectory)
        print(f"✅ {steps_count} trajectory steps saved in: {session_dir}")
        self._current_session_dir = session_dir
        return filepath

    def trajectory_html_viewer(self, output_dir: str = "trajectories", eval_score: Optional[float] = None) -> str:
        # Same viewer as OpenAI for parity
        if hasattr(self, '_current_session_dir') and self._current_session_dir:
            session_dir = self._current_session_dir
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uniq = uuid.uuid4().hex[:6]
            session_dir = os.path.join(output_dir, f"{self.name}_trajectory_{session_timestamp}-{uniq}")
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

        with open(html_path, 'w', encoding="utf-8") as f:
            f.write(html_content)

        print(f"📱 Open in browser: file://{os.path.abspath(html_path)}")
        return html_path

    def clear_trajectory(self):
        self._trajectory = []
        self._step_count = 0
        self._session_start_time = datetime.now().isoformat()
        if hasattr(self, "_current_session_dir"):
            delattr(self, "_current_session_dir")
        self._messages = []
        self._pending_tool_use_id = None
        self._pending_action_name = None
        print("🧹 Trajectory cleared")

    # --------------------------- helpers ---------------------------

    def _append_user_message(self, blocks: List[Dict[str, Any]]):
        self._messages.append({"role": "user", "content": blocks})

    def _append_assistant_message(self, blocks: Any):
        self._messages.append({"role": "assistant", "content": blocks})

    def _make_tool_result_block(self, tool_use_id: str, *, base64_png: Optional[str], note: str = "") -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        if note:
            blocks.append({"type": "text", "text": note})
        if base64_png:
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64_png}
            })
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": blocks}

    def rescale(self, obs: Observation) -> Observation:
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

    def translate(self, action_input: Dict[str, Any]) -> Action:
        """
        Translate Claude tool_use.input into a theta Action.
        """
        t = str(action_input.get("action", "")).lower()

        def map_xy_xylist(val) -> Tuple[float, float]:
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                x, y = val[0], val[1]
            elif isinstance(val, dict):
                x, y = val.get("x", 0), val.get("y", 0)
            else:
                x, y = 0, 0
            return float(x) * self._sx, float(y) * self._sy

        def map_coord() -> Tuple[float, float]:
            return map_xy_xylist(action_input.get("coordinate", [0, 0]))

        # Clicks (Claude supports more variants)
        if t in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            x, y = map_coord()
            button = "left"
            click_type = "click"
            if t == "right_click":
                button = "right"
            elif t == "middle_click":
                button = "middle"
            elif t == "double_click":
                click_type = "double_click"
            elif t == "triple_click":
                click_type = "triple_click"
            return ClickAction(type=click_type, point=Point(x=x, y=y), button=button)

        if t == "mouse_move":
            x, y = map_coord()
            return MoveAction(type="move", point=Point(x=x, y=y))

        if t == "type":
            return TypeAction(type="type_text", text=str(action_input.get("text", "")))

        if t == "key":
            combo = str(action_input.get("text", "") or action_input.get("keys", "") or action_input.get("key", ""))
            parts = [p.strip() for p in combo.split("+")] if combo else []
            keys = [Key(key=p) for p in parts if p]
            return KeyPressAction(type="key_press", keys=keys if keys else ([Key(key=combo)] if combo else []))

        # Scroll: direction + amount (wheel steps). No scaling of amount.
        if t == "scroll":
            x, y = map_coord()
            dx = dy = 0
            if "scroll_direction" in action_input and "scroll_amount" in action_input:
                direction = str(action_input.get("scroll_direction", "down")).lower()
                amt = int(action_input.get("scroll_amount", 1) or 1)
                if amt < 0:
                    amt = -amt
                if direction in ("up", "down"):
                    dy = +amt if direction == "up" else -amt
                elif direction in ("right", "left"):
                    dx = +amt if direction == "right" else -amt
            else:
                # Rare: pixel-like deltas; treat as wheel units directly (no scaling)
                dx = int(action_input.get("scroll_x", 0) or 0)
                dy = int(action_input.get("scroll_y", 0) or 0)
            return ScrollAction(type="scroll", point=Point(x=x, y=y), scroll_delta=Point(x=dx, y=dy))

        # Drag: left_click_drag (start_coordinate optional) -> path
        if t == "left_click_drag":
            pts: List[Point] = []

            def to_point(val):
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    xx, yy = float(val[0]), float(val[1])
                elif isinstance(val, dict):
                    xx, yy = float(val.get("x", 0)), float(val.get("y", 0))
                else:
                    return None
                return Point(x=xx * self._sx, y=yy * self._sy)

            if "start_coordinate" in action_input:
                p = to_point(action_input["start_coordinate"])
                if p:
                    pts.append(p)
            if "coordinate" in action_input:
                p = to_point(action_input["coordinate"])
                if p:
                    pts.append(p)
            if not pts and "path" in action_input and isinstance(action_input["path"], (list, tuple)):
                for node in action_input["path"]:
                    p = to_point(node)
                    if p:
                        pts.append(p)
            if not pts:
                raise NotImplementedError("left_click_drag requires at least an end coordinate.")
            return DragAction(type="drag", path=pts)

        # Fine-grained mouse + hold (Claude-only)
        if t in ("left_mouse_down", "mouse_down"):
            x, y = map_coord()
            return MouseDownAction(button="left", point=Point(x=x, y=y))
        if t in ("left_mouse_up", "mouse_up"):
            x, y = map_coord()
            return MouseUpAction(button="left", point=Point(x=x, y=y))
        if t == "hold_key":
            combo = str(action_input.get("text", "") or action_input.get("key", "")).strip()
            parts = [p.strip() for p in combo.split("+")] if combo else []
            keys = [Key(key=p) for p in parts if p]
            dur = float(action_input.get("duration", 0.5))
            return HoldKeyAction(keys=keys, duration=dur)

        if t in ("screenshot", "wait"):
            # handled inline in act() by sending tool_result with a fresh image
            raise NotImplementedError(f"Internal-only action surfaced to translate(): {t}")

        # OpenAI-specific action shape not valid here
        if t == "drag":
            raise NotImplementedError("Unsupported Claude action: drag")

        raise NotImplementedError(f"Unhandled action type: {t}")

    # --------------------------- utility ---------------------------

    def _collect_thinking_blocks(self, content: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in content or []:
            t = self._safe_attr(block, "type")
            if t in ("thinking", "redacted_thinking"):
                if isinstance(block, dict):
                    out.append(block)
                elif hasattr(block, "model_dump"):
                    out.append(block.model_dump())
                elif hasattr(block, "__dict__"):
                    out.append({k: getattr(block, k) for k in block.__dict__.keys()})
                else:
                    out.append({"type": t, "raw": str(block)})
        return out

    def _join_thinking_text(self, blocks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for b in blocks:
            if b.get("type") == "thinking":
                txt = b.get("thinking") or b.get("text") or b.get("content")
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts)

    def _append_user_message(self, blocks: List[Dict[str, Any]]):
        self._messages.append({"role": "user", "content": blocks})

    def _append_assistant_message(self, blocks: Any):
        self._messages.append({"role": "assistant", "content": blocks})

    def _safe_attr(self, obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _serialize_object(self, obj: Any) -> Any:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_object(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_object(v) for k, v in obj.items()}
        elif hasattr(obj, "__dict__"):
            return {k: self._serialize_object(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        else:
            return str(obj)

    def _log_step(self, step_data: Dict[str, Any]) -> None:
        if not getattr(self, "_enable_logging", False):
            return
        try:
            step = step_data.get("step")
            
            reasoning = step_data.get("reasoning")
            assistant_text = step_data.get("assistant_text")
            tool_call_raw = step_data.get("tool_call_raw")
            action_taken = step_data.get("action_taken")

            lines: List[str] = []
            lines.append(f"Step {step}")
            lines.append("------------------------")
            lines.append("Reasoning:")
            lines.append(reasoning or "None")
            lines.append("------------------------")
            lines.append("Text Output:")
            lines.append(assistant_text or "None")
            lines.append("------------------------")
            lines.append("Computer Tool Call:")
            lines.append(json.dumps(tool_call_raw, indent=2, ensure_ascii=False) if tool_call_raw is not None else "None")
            lines.append("------------------------")
            lines.append("Translated Action:")
            lines.append(json.dumps(action_taken, indent=2, ensure_ascii=False) if action_taken is not None else "None")
            lines.append("=====================")

            self.logger.info("\n".join(lines))
        except Exception:
            pass
