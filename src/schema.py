from typing import List, Optional
from pydantic import BaseModel, Field


class ImageVisionMetadata(BaseModel):
    subject: str = Field(..., description="The main subject of the image (e.g. 'red fox', 'gray wolf', 'golden retriever')")
    category: str = Field(..., description="Broad category (e.g. 'animal', 'landscape', 'vehicle')")
    attributes: List[str] = Field(default_factory=list, description="Descriptive visual tags/attributes")
    caption: str = Field(..., description="A concise factual description of what is depicted in the image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class ImageRecordResponse(BaseModel):
    id: int
    filename: str
    subject: Optional[str] = None
    category: Optional[str] = None
    attributes: List[str] = []
    caption: Optional[str] = None
    confidence: Optional[float] = None
    is_flagged: bool
    status: str
