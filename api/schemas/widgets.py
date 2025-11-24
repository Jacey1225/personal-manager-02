from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Any
import uuid
from enum import Enum

class WidgetSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

class WidgetInteraction(BaseModel):
    endpoint: str = Field(default="", description="The API endpoint for the widget interaction")
    params: dict = Field(default_factory=dict, description="The parameters for the widget interaction")
    headers: dict = Field(default_factory=dict, description="The headers for the widget interaction")
    refresh_interval: int = Field(default=0, description="The refresh interval for the widget interaction in seconds")

class MediaFile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Server file path")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type")
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail")
    duration: Optional[float] = Field(None, description="Duration for video/audio files")
    dimensions: Optional[dict[str, int]] = Field(None, description="Width/height for images/videos")
    upload_timestamp: str = Field(..., description="Upload timestamp")

class VideoFile(MediaFile):
    duration: float = Field(..., description="Video duration in seconds")
    dimensions: Dict[str, int] = Field(..., description="Video dimensions (width, height)")
    fps: Optional[float] = Field(None, description="Frames per second")
    bitrate: Optional[int] = Field(None, description="Video bitrate in kbps")
    codec: Optional[str] = Field(None, description="Video codec used")
    audio_codec: Optional[str] = Field(None, description="Audio codec used")
    has_audio: bool = Field(default=True, description="Whether video has audio track")
    chapters: Optional[List[Dict[str, Any]]] = Field(None, description="Video chapters if available")
    subtitles: Optional[List[str]] = Field(None, description="Available subtitle languages")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "presentation.mp4",
                "duration": 1250.5,
                "dimensions": {"width": 1920, "height": 1080},
                "fps": 30.0,
                "codec": "h264",
                "audio_codec": "aac"
            }
        }

class AudioFile(MediaFile):
    duration: float = Field(..., description="Audio duration in seconds")
    bitrate: Optional[int] = Field(None, description="Audio bitrate in kbps")
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz")
    channels: Optional[int] = Field(None, description="Number of audio channels")
    codec: Optional[str] = Field(None, description="Audio codec used")
    
    # Metadata fields for music files
    title: Optional[str] = Field(None, description="Track title")
    artist: Optional[str] = Field(None, description="Artist name")
    album: Optional[str] = Field(None, description="Album name")
    genre: Optional[str] = Field(None, description="Music genre")
    year: Optional[int] = Field(None, description="Release year")
    track_number: Optional[int] = Field(None, description="Track number")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "meeting_recording.mp3",
                "duration": 3600.0,
                "bitrate": 320,
                "sample_rate": 44100,
                "channels": 2,
                "codec": "mp3"
            }
        }

class ImageFile(MediaFile):
    dimensions: Dict[str, int] = Field(..., description="Image dimensions (width, height)")
    color_mode: Optional[str] = Field(None, description="Color mode (RGB, RGBA, CMYK, etc.)")
    dpi: Optional[int] = Field(None, description="Dots per inch")
    has_transparency: bool = Field(default=False, description="Whether image has transparency")
    exif_data: Optional[Dict[str, Any]] = Field(None, description="EXIF metadata for photos")
    dominant_colors: Optional[List[str]] = Field(None, description="Dominant colors in hex format")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "project_logo.png",
                "dimensions": {"width": 1024, "height": 768},
                "color_mode": "RGBA",
                "has_transparency": True,
                "dominant_colors": ["#FF5733", "#33FF57", "#3357FF"]
            }
        }

class PDFFile(MediaFile):
    page_count: int = Field(..., description="Number of pages in PDF")
    is_encrypted: bool = Field(default=False, description="Whether PDF is password protected")
    has_forms: bool = Field(default=False, description="Whether PDF contains fillable forms")
    has_bookmarks: bool = Field(default=False, description="Whether PDF has bookmarks/outline")
    text_extractable: bool = Field(default=True, description="Whether text can be extracted")
    
    # Content analysis
    extracted_text: Optional[str] = Field(None, description="Extracted text content")
    text_preview: Optional[str] = Field(None, description="First 500 characters of text")
    language: Optional[str] = Field(None, description="Detected language")
    word_count: Optional[int] = Field(None, description="Approximate word count")
    
    # PDF metadata
    title: Optional[str] = Field(None, description="PDF title from metadata")
    author: Optional[str] = Field(None, description="PDF author from metadata")
    subject: Optional[str] = Field(None, description="PDF subject from metadata")
    keywords: Optional[List[str]] = Field(None, description="PDF keywords")
    creation_date: Optional[str] = Field(None, description="PDF creation date")
    modification_date: Optional[str] = Field(None, description="PDF last modification date")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "project_proposal.pdf",
                "page_count": 25,
                "text_extractable": True,
                "word_count": 5420,
                "language": "en",
                "title": "Project Proposal 2025"
            }
        }

class CSVFile(MediaFile):
    row_count: int = Field(..., description="Number of rows (excluding header)")
    column_count: int = Field(..., description="Number of columns")
    has_header: bool = Field(default=True, description="Whether first row is header")
    delimiter: str = Field(default=",", description="CSV delimiter character")
    encoding: str = Field(default="utf-8", description="File encoding")
    
    # Column information
    column_names: List[str] = Field(..., description="Column names/headers")
    column_types: Dict[str, str] = Field(..., description="Detected data types for each column")
    column_stats: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Basic statistics for numeric columns")
    
    # Data preview
    sample_rows: List[List[str]] = Field(..., description="First 5 rows of data")
    data_quality: Dict[str, Any] = Field(
        default_factory=lambda: {
            "null_values": 0,
            "duplicate_rows": 0,
            "completeness_percentage": 100.0
        },
        description="Data quality metrics"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "sales_data.csv",
                "row_count": 1500,
                "column_count": 8,
                "column_names": ["date", "product", "price", "quantity"],
                "column_types": {"date": "datetime", "product": "string", "price": "float", "quantity": "integer"}
            }
        }

class DocumentFile(MediaFile):
    """For Word documents, text files, etc."""
    page_count: Optional[int] = Field(None, description="Number of pages (for paginated docs)")
    word_count: Optional[int] = Field(None, description="Word count")
    character_count: Optional[int] = Field(None, description="Character count")
    paragraph_count: Optional[int] = Field(None, description="Number of paragraphs")
    
    # Content
    extracted_text: Optional[str] = Field(None, description="Extracted text content")
    text_preview: Optional[str] = Field(None, description="First 500 characters")
    language: Optional[str] = Field(None, description="Detected language")
    
    # Document metadata
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    keywords: Optional[List[str]] = Field(None, description="Document keywords")
    creation_date: Optional[str] = Field(None, description="Creation date")
    modification_date: Optional[str] = Field(None, description="Last modification date")
    
    # Structure analysis
    has_images: bool = Field(default=False, description="Contains images")
    has_tables: bool = Field(default=False, description="Contains tables")
    has_links: bool = Field(default=False, description="Contains hyperlinks")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "meeting_notes.docx",
                "word_count": 1200,
                "paragraph_count": 15,
                "language": "en",
                "has_tables": True
            }
        }

class ArchiveFile(MediaFile):
    """For ZIP, RAR, TAR files, etc."""
    file_count: int = Field(..., description="Number of files in archive")
    uncompressed_size: int = Field(..., description="Total uncompressed size in bytes")
    compression_ratio: float = Field(..., description="Compression ratio (0.0 to 1.0)")
    is_encrypted: bool = Field(default=False, description="Whether archive is password protected")
    
    # Archive contents
    file_list: List[Dict[str, Any]] = Field(..., description="List of files in archive")
    directory_structure: Dict[str, Any] = Field(..., description="Directory tree structure")
    file_types_summary: Dict[str, int] = Field(..., description="Count of each file type")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "project_assets.zip",
                "file_count": 45,
                "uncompressed_size": 52428800,
                "compression_ratio": 0.65,
                "file_types_summary": {"jpg": 20, "png": 15, "txt": 10}
            }
        }

class SpreadsheetFile(MediaFile):
    """For Excel, LibreOffice Calc files"""
    sheet_count: int = Field(..., description="Number of worksheets")
    sheet_names: List[str] = Field(..., description="Names of all worksheets")
    total_rows: int = Field(..., description="Total rows across all sheets")
    total_columns: int = Field(..., description="Total columns across all sheets")
    
    # Per-sheet information
    sheet_info: List[Dict[str, Any]] = Field(..., description="Information about each sheet")
    has_formulas: bool = Field(default=False, description="Contains Excel formulas")
    has_charts: bool = Field(default=False, description="Contains charts/graphs")
    has_pivot_tables: bool = Field(default=False, description="Contains pivot tables")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "financial_report.xlsx",
                "sheet_count": 3,
                "sheet_names": ["Summary", "Q1 Data", "Q2 Data"],
                "has_formulas": True,
                "has_charts": True
            }
        }

class CodeLogic(BaseModel):
    id: str = Field(..., description="The unique identifier for the code logic")
    media_id: str = Field(..., description="The unique identifier for the media connected to this logic")
    endpoint: str = Field(..., description="The API endpoint for the code logic")
    method: str = Field(..., description="The HTTP method (GET, POST, etc.)")
    headers: Dict[str, str] = Field(default_factory=dict, description="Headers to include in the request")
    code: str = Field(..., description="The code to execute")

class WidgetContent(BaseModel):
    text: Optional[str] = Field(default=None, description="If the widget has text content")
    image: Optional[MediaFile] = Field(default=None, description="If the widget has an image")
    video: Optional[VideoFile] = Field(default=None, description="If the widget has a video")
    audio: Optional[AudioFile] = Field(default=None, description="If the widget has audio content")
    document: Optional[PDFFile] = Field(default=None, description="If the widget has a document")
    link: Optional[str] = Field(default=None, description="If the widget has a link")
    chart: Optional[Union[CSVFile, DocumentFile, SpreadsheetFile, ArchiveFile]] = Field(default=None, description="If the widget has a chart")
    code: Optional[list[CodeLogic]]

class WidgetConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the widget")
    name: str = Field(default="", description="The name of the widget")
    description: Optional[str] = Field(default=None, description="A brief description of the widget")
    widget_type: WidgetContent = Field(description="The type of the widget -> ('chart', 'gallery', 'text', 'document', 'video', 'form', etc)")
    size: WidgetSize = Field(default=WidgetSize.SMALL, description="The size of the widget")
    interaction: WidgetInteraction = Field(default_factory=WidgetInteraction, description="The interaction configuration for the widget")


