from fastapi import APIRouter, HTTPException
from api.config.strict_pyenv import validate_code, ValidateRunTime
from api.config.fetchMongo import MongoHandler
from api.schemas.auth import User
from api.schemas.widgets import WidgetInteractionRequest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

public_router = APIRouter()
project_config = MongoHandler("userAuthDatabase", "openProjects")
widget_config = MongoHandler("userAuthDatabase", "openWidgets")

@public_router.post("/public/startup")
async def public_startup(
    widget_interaction: WidgetInteractionRequest
):
    logger.info(f"Received public startup request for project: {widget_interaction.project_id}")
    project = await project_config.get_single_doc({"project_id": widget_interaction.project_id})
    logger.info(f"Project retrieved: {project}")
    if project["widgets"] and isinstance(project["widgets"], list):
        widgets_info = []
        for widget_id in project["widgets"]:
            widget = await widget_config.get_single_doc({"widget_id": widget_id})
        return {"status": "success", "widgets": widgets_info}


@public_router.post("/public")
async def public_post(
    widget_interaction: WidgetInteractionRequest
):
    logger.info(f"Received public widget interaction request for project: {widget_interaction.project_id}, widget: {widget_interaction.widget_id}, endpoint: {widget_interaction.endpoint}")
    project = await project_config.get_single_doc({"project_id": widget_interaction.project_id})
    logger.info(f"Project retrieved: {project}")
    widget = await widget_config.get_single_doc({"widget_id": widget_interaction.widget_id})
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    logger.info(f"Widget retrieved: {widget}")

    code_logic = widget["interactions"][widget_interaction.endpoint]["logic"]
    context = widget_interaction.params

    validate_code(code_logic)
    executor = ValidateRunTime(code_logic)
    result = executor.run(code_logic, context)
    
    return {"status": "success", "result": result}

    