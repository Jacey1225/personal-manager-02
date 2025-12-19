from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from enum import Enum

class WidgetSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

class WidgetComponents(BaseModel): 
    type: str = Field(default="", description="The type of the widget component, eg. 'button', 'input', etc.")
    content: List[Any] = Field(default=[], description="The content of the widget component")
    properties: Dict[str, Any] = Field(default_factory=dict, description="The properties of the widget component, eg. {'color': 'blue', 'size': 'medium'}")

class WidgetInteraction(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict, description="The parameters for the widget interaction")
    headers: Dict[str, str] = Field(default_factory=dict, description="The headers for the widget interaction")
    refresh_interval: int = Field(default=0, description="The refresh interval for the widget interaction in seconds")
    components: List[WidgetComponents] = Field(default_factory=list, description="The components of the widget interaction")
    logic: str = Field(default="", description="The logic for the widget interaction")

class WidgetConfig(BaseModel):
    widget_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the widget")
    name: str = Field(default="", description="The name of the widget")
    size: WidgetSize = Field(default=WidgetSize.SMALL, description="The size of the widget")
    interactions: Dict[str, WidgetInteraction] = Field(default_factory=dict, 
                                                       description="The interaction configuration for the widget (endpoint, logic)")

class WidgetInteractionRequest(BaseModel):
    user_id: str
    project_id: str
    widget_id: str
    endpoint: str
    headers: Dict[str, str]
    params: dict
    
