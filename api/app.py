from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import logging

app = FastAPI()
schedule_router = APIRouter(prefix="/scheduler", tags=["scheduler"])
app.include_router(schedule_router)


@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)