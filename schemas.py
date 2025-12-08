from pydantic import BaseModel, Field
from typing import Literal

class GenerationResponse(BaseModel):
    status: Literal["fail"] = Field(..., example="fail")
    detail: str = Field(..., example="Возникла ошибка при обработке изображения")