from fastapi import FastAPI
from api.routes.auth.user import auth_router
from api.routes.auth.google import google_router
from api.routes.auth.public import oauth_router
from api.routes.public_router import public_router
from api.routes.tasklist_router import task_list_router
from api.routes.project_router import project_router
from api.routes.organization_router import organization_router
from api.routes.event_router import event_router
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler 
import uvicorn
from datetime import datetime, timezone
import pytz

scheduler = AsyncIOScheduler()

def build_app() -> FastAPI:
    app = FastAPI(
        title="Personal Manager Agent",
        description="An agent for managing personal tasks and events.",
        version="1.1.0"
    )

    # More restrictive CORS for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.include_router(event_router)
    app.include_router(auth_router)
    app.include_router(google_router)
    app.include_router(oauth_router)
    app.include_router(public_router)
    app.include_router(task_list_router)
    app.include_router(project_router)
    app.include_router(organization_router)

    return app

async def midnight_refresh():
    utc_now = datetime.now(timezone.utc)
    local_tz = pytz.timezone('America/Los_Angeles')  # Change to your local timezone
    local_now = utc_now.astimezone(local_tz)
    print(f"Midnight refresh executed at {local_now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")


app = build_app()
if __name__ == "__main__":
    print(f"Starting FastAPI app...")
    uvicorn.run(
        "api.main:app", 
        host="0.0.0.0", 
        port=8000)