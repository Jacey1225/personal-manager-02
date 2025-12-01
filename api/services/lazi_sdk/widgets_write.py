from api.config.fetchMongo import MongoHandler
from api.config.s3 import S3Handler
from api.routes.auth.public import OAuthUser
import logging
from fastapi import HTTPException
from api.schemas.widgets import WidgetConfig, WidgetSize, WidgetInteraction, WidgetComponents
from api.schemas.auth import Scopes
from typing import Optional, Annotated, Any, Callable
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
project_config = MongoHandler("userAuthDatabase", "openProjects")
widget_config = MongoHandler("userAuthDatabase", "openWidgets")

class WriteWidget:
    def __init__(self, 
                 username: str, 
                 token: str,
                 project_id: str):
        self.username = username
        self.token = token
        self.project_id = project_id
        self.current_widget = WidgetConfig()
        self.s3_client = S3Handler()
        self.oauth_client = OAuthUser(username, token)

    async def create(
            self,
            name: Optional[str]=None,
            size: Optional[str]=None,
    ):
        try:
            self.current_widget = WidgetConfig(
                name=name if name else "",
                size=WidgetSize(size) if size else WidgetSize.SMALL,
            )
            return True
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def interaction(self,
             endpoint: str, 
             headers: dict,
             refresh_interval: int,
             func: Callable):
        code_logic: str = inspect.getsource(func)
        interaction = WidgetInteraction(
            headers=headers,
            refresh_interval=refresh_interval,
            logic=code_logic
        )
        if endpoint not in self.current_widget.interactions:
            self.current_widget.interactions[endpoint] = interaction
        
        return self.current_widget.interactions[endpoint]
    
    async def component(
            self,
            endpoint: str,
            type: str,
            content: list[Any],
            props: dict[str, Any]
    ):
        component = WidgetComponents(
            type=type,
            content=content,
            properties=props
        )
        if endpoint not in self.current_widget.interactions:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        self.current_widget.interactions[endpoint].components.append(component)
        return self.current_widget.interactions[endpoint].components[-1]
    
    async def uploadable(
            self,
            object_name: str,
            expire: int,
            filename: Optional[str]
    ):
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 client not initialized")
        
        media_object = self.s3_client.fetch_file(
            object_name
        )
        if not media_object and filename:
            await self.s3_client.upload_file(
                filename,
                object_name
            )
        presigned_url = self.s3_client.generate_presigned_url(
            object_name,
            expire
        )
        return presigned_url

    async def save(self):
        auth = await self.oauth_client.get_active_user()
        if not auth or Scopes.WIDGETS_WRITE not in auth["payload"]["scopes"]:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        await widget_config.post_insert(self.current_widget.model_dump())
        project = await project_config.get_single_doc({"project_id": self.project_id})
        if not project:
            raise Exception("Project not found")

        project["widgets"].append(self.current_widget.id)
        await project_config.post_update({"project_id": self.project_id}, {"$set": {"widgets": project["widgets"]}})

        logger.info(f"Widget {self.current_widget.id} saved to project {self.project_id}")
        await widget_config.close_client()
        await project_config.close_client()
