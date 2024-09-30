from pydantic import BaseModel
from typing import List, Optional

class TrainResponse(BaseModel):
    AUPRC: Optional[float] = None
    y_test: Optional[List[int]] = None
    y_pred: Optional[List[int]] = None
    status: str
    task_id: Optional[str] = None

class VizualizationResponse(BaseModel):
    image_data: str #Base64 encoded image

class StatusResponse(BaseModel):
    task_id : str
    status: str
    AUPRC: Optional[float] = None
    y_test: Optional[List[int]] = None
    y_pred: Optional[List[int]] = None