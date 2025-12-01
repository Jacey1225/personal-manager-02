from api.config.fetchMongo import MongoHandler
from api.config.s3 import S3Handler
from api.schemas.auth import Scopes, UserInDB
from fastapi import HTTPException
from typing import Optional, Annotated, Any, Callable
import logging
from api.routes.auth.public import OAuthUser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

projects_config = MongoHandler("userAuthDatabase", "openProjects")
widgets_config = MongoHandler("userAuthDatabase", "openWidgets")

class ReadWidget:
    def __init__(self,
                 username: str,
                 token: str,
                 project_id: str):
        self.username = username
        self.token = token
        self.project_id = project_id

        self.oauth = OAuthUser(username, token)
        self.s3_client = S3Handler()

    async def list_widgets(self):
        auth = await self.oauth.get_active_user()  
        if not auth or Scopes.WIDGETS_READ not in auth["payload"]["scopes"]:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        project = await projects_config.get_single_doc({"project_id": self.project_id})
        widget_ids = project["widgets"]
        widgets = []
        for widget_id in widget_ids:
            widget = await widgets_config.get_single_doc({"widget_id": widget_id})
            if widget:
                widgets.append({"widget_id": widget["widget_id"], "name": widget["name"]})
        
        await projects_config.close_client()
        await widgets_config.close_client()
        return widgets
    
    async def get_widget(self, widget_id: str):
        auth = await self.oauth.get_active_user()
        if not auth or Scopes.WIDGETS_READ not in auth["payload"]["scopes"]:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        widget = await widgets_config.get_single_doc({"widget_id": widget_id})
        if not widget:
            raise HTTPException(status_code=404, detail="Widget not found")
        
        await widgets_config.close_client()
        return widget