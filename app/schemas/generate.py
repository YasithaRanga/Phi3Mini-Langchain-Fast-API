# app/schemas/generate.py
from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
