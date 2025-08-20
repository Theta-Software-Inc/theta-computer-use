import uuid
from enum import Enum
from typing import Optional, Literal, Union, List, Annotated
from pydantic import BaseModel, Field


class CreateEnvRequest(BaseModel):
    session: str
    task_ids: List[str]

class Observation(BaseModel):
    screenshot: str
    text: Optional[str] = None

class EnvStatus(str, Enum):
    RUNNING = "running"
    CLOSED = "closed"
    ERROR = "error"

class Run(BaseModel):
    score: float
    steps: int
    task_id: str

class EnvRef(BaseModel):
    env_id: uuid.UUID
    status: EnvStatus




class StepEnvResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# Action models
class Point(BaseModel):
    """Absolute or normalized coordinates depending on API usage context."""
    x: float
    y: float


class Key(BaseModel):
    key: str


class ClickAction(BaseModel):
    type: Literal["click", "double_click"] = "click"
    point: Optional[Point] = None
    button: Literal["left", "right", "middle"] = "left"


class ScrollAction(BaseModel):
    type: Literal["scroll"] = Field(default="scroll")
    point: Optional[Point] = None
    scroll_delta: Point


class MoveAction(BaseModel):
    type: Literal["move"] = Field(default="move")
    point: Point


class DragAction(BaseModel):
    type: Literal["drag"] = Field(default="drag")
    path: List[Point]


class TypeAction(BaseModel):
    type: Literal["type_text"] = Field(default="type_text")
    text: str


class KeyPressAction(BaseModel):
    type: Literal["key_press"] = Field(default="key_press")
    keys: List[Key]


Action = Annotated[
    Union[
        ClickAction,
        ScrollAction,
        MoveAction,
        DragAction,
        TypeAction,
        KeyPressAction,
    ],
    Field(discriminator="type"),
]
