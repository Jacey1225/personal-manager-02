from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
from datetime import datetime
    
class PageSchema(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the page")
    project_id: str = Field(..., description="The ID of the project this page belongs to")
    name: str = Field(default="", description="The name of the page")
    author_id: str = Field(description="The User ID of the page author")
    contributors: List[str] = Field(default=[], description="List of User IDs who contributed to the page")
    creation_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="The timestamp when the page was created")
    page_type: str = Field(description="The type of the page (e.g., 'discussion', 'gallery', etc.)")
    interface_type: str = Field(description="The interface type of the page (e.g., 'file_upload', 'chat', etc.)")
    api_config: dict = Field(default_factory=dict, description="API configuration for the page")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the page")

class ProjectDetails(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_id: str = Field(..., description="Unique identifier for the project")
    project_likes: int = Field(default=0, description="Number of likes for the project")
    project_transparency: bool = Field(default=True, description="Transparency status of the project(True: public - False: private)")
    project_members: List[str] = Field(..., description="List of user IDs associated with the project")
    organizations: Optional[List[str]] = Field(default=[], description="Organization IDs associated with the project")
class Organization(BaseModel):
    name: str = Field(description="The name of the organization")
    members: List[str] = Field(description="List of user IDs associated with the organization")
    id: str = Field(default=str(uuid.uuid4()), description="Unique identifier for the organization")
    projects: List[str] = Field(default_factory=list, description="List of project IDs associated with the organization")
class DiscussionRequest(BaseModel):
    user_id: str
    project_id: str
    force_refresh: bool

    def __hash__(self):
        return hash((self.user_id, self.project_id, self.force_refresh))

    def __eq__(self, other):
        if not isinstance(other, DiscussionRequest):
            return False
        return (self.user_id, self.project_id, self.force_refresh) == \
               (other.user_id, other.project_id, other.force_refresh)
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