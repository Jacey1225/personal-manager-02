from fastapi import FastAPI, Request
import aiohttp
import json
from slack_bolt import App
from slack_sdk import WebClient
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.model_setup.structure_model_output import EventDetails
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from api.commandline.main_model import MainModel
from dotenv import load_dotenv
import os

app = App(token=os.getenv("SLACK_AUTH"))
handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
client = WebClient(token=os.getenv("SLACK_AUTH"))
fastapi_app = FastAPI()    

class SlackMessage(BaseModel):
    channel: str = Field(..., description="The Slack channel ID")
    text: str = Field(..., description="The message text")
    email: str = Field(..., description="The user's email address")
    username: str = Field(..., description="The user's username")
    user_id: str = Field(..., description="The user's ID")

async def post_to_slack(channel: str, text: str):
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {os.getenv('SLACK_AUTH')}",
            "Content-Type": "application/json"
        }
        body = {
            "channel": channel,
            "text": text
        }
        async with session.post("https://slack.com/api/chat.postMessage", json=body, headers=headers) as response:
            return await response.json()
        
async def get_user_from_cache(slack_id: str) -> Optional[Dict[str, Any]]:
    with open(f"data/slack_cache", "r") as f:
        cache = json.load(f)
    
    if slack_id in cache:
        user_id = cache[slack_id]
        with open(f"data/users/{user_id}.json", "r") as f:
            user_info = json.load(f)
            return user_info
    return None
        
async def get_user_info(slack_id: str) -> Optional[Dict[str, Any]]:
    response = client.users_info(user=slack_id)
    found_id = ""
    if response["ok"]:
        user_data = response["user"]
        if user_data:
            email = user_data.get("profile", {}).get("email")
            
            with open(f"data/user_log.json", "r") as f:
                user_log = json.load(f)
                for user_info, user_id in user_log.items():
                    if email in user_info:
                        found_id = user_id
                        break

    if found_id is not "":
        with open(f"data/users/{found_id}.json", "r") as f:
            user_info = json.load(f)
            return user_info
        with open(f"data/slack_cache.json", "w") as f:
            json.dump({slack_id: found_id}, f)

    return None


