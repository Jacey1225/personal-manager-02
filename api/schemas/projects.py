from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uuid
from datetime import datetime

class DiscussionRequest(BaseModel):
    user_id: str
    project_id: str


class DiscussionData(BaseModel):
    title: str = Field(description="The title of the discussion")
    author_id: str = Field(description="The User ID of the discussion author")
    active_contributors: List[str] = Field(description="The usernames of everyone who currently contributes to this discussion")
    content: List[Dict[str, str]] = Field(description="A list of message objects with username, message, and timestamp")
    created_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="The timestamp when the discussion was created")
    transparency: bool = Field(description="The visibility status of the discussion (e.g., True, False)")

class Discussion(BaseModel):
    discussion_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The UUID of the discussion")
    project_id: str = Field(default="", description="The ID of the project this discussion is linked to, if any")
    data: DiscussionData

class Organization(BaseModel):
    name: str = Field(description="The name of the organization")
    members: List[str] = Field(description="List of user IDs associated with the organization")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the organization")
    projects: List[str] = Field(default_factory=list, description="List of project IDs associated with the organization")

class CreateOrgRequest(BaseModel):
    id: str
    name: str
    members: list[str]
    projects: list[str]

class OrgRequest(BaseModel):
    user_id: str
    organization_id: str

class ResourceDetails(BaseModel):
    resource_id: str = Field(default=str(uuid.uuid4()), description="Unique identifier for the resource")
    resource_name: str = Field(..., description="Name of the resource")
    resource_link: str = Field(..., description="Link to the resource (e.g., document, link, sheet)")
    resource_owner: str = Field(..., description="User ID of the resource owner")
    resource_timestamp: str = Field(default=datetime.now().isoformat(), description="Timestamp when the resource was created or modified")

class ProjectDetails(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_id: str = Field(..., description="Unique identifier for the project")
    project_likes: int = Field(default=0, description="Number of likes for the project")
    project_transparency: bool = Field(default=True, description="Transparency status of the project(True: public - False: private)")
    project_members: List[str] = Field(..., description="List of user IDs associated with the project")
    organizations: Optional[List[str]] = Field(default=[], description="Organization IDs associated with the project")

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