from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
from datetime import datetime
from api.schemas.widgets import WidgetConfig
    
class ProjectDetails(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_id: str = Field(..., description="Unique identifier for the project")
    widgets: List[str] = Field(default=[], description="List of widgets associated with the project")
    project_likes: int = Field(default=0, description="Number of likes for the project")
    project_transparency: bool = Field(default=True, description="Transparency status of the project(True: public - False: private)")
    project_members: List[str] = Field(..., description="List of user IDs associated with the project")
    organizations: List[str] = Field(default=[], description="Organization IDs associated with the project")

class PageSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the page")
    project_name: str = Field(..., description="The name of the project this page belongs to")
    widget_contents: List[dict] = Field(default=[], description="List of widget contents associated with the page")
    project_likes: int = Field(default=0, description="Number of likes for the project this page belongs to")
    project_transparency: bool = Field(default=True, description="Transparency status of the project this page belongs to")
    project_members: List[tuple[str, str]] = Field(default=[("", "")], description="The members of the project this page belongs to")
    organizations: List[str] = Field(default=[], description="List of organization IDs associated with the project this page belongs to")

class Organization(BaseModel):
    name: str = Field(description="The name of the organization")
    members: List[str] = Field(description="List of user IDs associated with the organization")
    id: str = Field(default=str(uuid.uuid4()), description="Unique identifier for the organization")
    projects: List[str] = Field(default_factory=list, description="List of project IDs associated with the organization")

class CreateOrgRequest(BaseModel):
    id: str
    name: str
    members: list[str]
    projects: list[str]

class OrgRequest(BaseModel):
    user_id: str
    organization_id: str
    force_refresh: bool

    def __hash__(self):
        return hash((self.user_id, self.organization_id, self.force_refresh))

    def __eq__(self, other):
        if not isinstance(other, OrgRequest):
            return False
        return (self.user_id, self.organization_id, self.force_refresh) == \
               (other.user_id, other.organization_id, other.force_refresh)
class CreateProjectRequest(BaseModel):
    project_name: str
    project_transparency: bool
    project_likes: int
    project_members: List[tuple[str ,str]] 
    user_id: str

class ModifyProjectRequest(BaseModel):
    project_id: str
    user_id: str
    project_name: str
    force_refresh: bool = False
    
    def __hash__(self):
        return hash((self.project_id, self.user_id, self.project_name, self.force_refresh))
    
    def __eq__(self, other):
        if not isinstance(other, ModifyProjectRequest):
            return False
        return (self.project_id, self.user_id, self.project_name, self.force_refresh) == \
               (other.project_id, other.user_id, other.project_name, other.force_refresh)