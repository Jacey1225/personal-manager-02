from fastapi import APIRouter, HTTPException
from api.config.strict_pyenv import validate_code, ValidateRunTime
from api.config.fetchMongo import MongoHandler
from api.schemas.auth import User
from api.schemas.widgets import WidgetInteractionRequest

public_router = APIRouter()
project_config = MongoHandler("userAuthDatabase", "openProjects")
widget_config = MongoHandler("userAuthDatabase", "openWidgets")


@public_router.post("/public")
async def public_post(
    widget_interaction: WidgetInteractionRequest
):
    project = await project_config.get_single_doc({"project_id": widget_interaction.project_id})
    if widget_interaction.widget_id not in project["widgets"]:
        raise HTTPException(status_code=404, detail="Widget not found in project")
    widget = await widget_config.get_single_doc({"widget_id": widget_interaction.widget_id})
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    
    code_logic = widget["interactions"][widget_interaction.endpoint]["logic"]
    context = widget_interaction.params

    validate_code(code_logic)
    executor = ValidateRunTime(code_logic)
    executor.run(code_logic, context)

    