from pydantic import BaseModel, Field
from typing import Literal

class GenerationResponse(BaseModel):
    status: Literal["ok"] = Field(..., example="ok")
    detail: str = Field(..., example="Изображение успешно преобразовано в аниме стиль")