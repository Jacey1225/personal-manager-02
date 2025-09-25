import pandas as pd
import os
import json
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from src.fetchMongo import MongoHandler

class SpreadSheet(BaseModel):
    file_path: str = Field(..., description="The path to the spreadsheet file")
    sheet_name: str = Field(default="Sheet1", description="The name of the sheet to read from the spreadsheet")
    columns: list[str] = Field(default_factory=list, description="The columns to read from the spreadsheet")
    rows: list[str] = Field(default=[], description="The rows to read from the spreadsheet")

    def convert_to_dict(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Spreadsheet file not found at path: {self.file_path}")
        
        return {
            self.sheet_name: {
                "columns": self.columns,
                "rows": self.rows,
            }}