import logging
from typing import Optional, Any, List, Dict, BinaryIO
import uuid
from datetime import datetime
import pandas as pd
import io
import os
import fitz
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import mimetypes
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from api.config.fetchMongo import MongoHandler
from api.schemas.widgets import (MediaFile, VideoFile, AudioFile, ImageFile, PDFFile, CSVFile, DocumentFile, ArchiveFile)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandleResources:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.file_dir = f"/uploads/{self.user_id}"