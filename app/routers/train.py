from fastapi import APIRouter, BackgroundTasks, Depends
from app.services import fraud_detection_service
from app.models.schemas import TrainResponse, StatusResponse
from modeling import FraudDetection
import uuid
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

#store task statuses (in-memory example)
task_statuses = {}

@router.post("/isolation_forest", response_model = TrainResponse)
async def train_isolation_forest(background_tasks : BackgroundTasks, fraud_detector : FraudDetection = Depends(fraud_detection_service.get_fraud_detector)):
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status":"started"}

    def training_task():
        try:
            result = fraud_detection_service.train_isolation_forest(fraud_detector)
            task_statuses[task_id] = {
                "status":"completed",
                "AUPRC":result.get('AUPRC'),
                "y_test":result.get('y_test'),
                "y_pred":result.get('y_pred')
            }
        except Exception as e:
            logger.error(f"Error during isolation forest training: {str(e)}")
            task_statuses[task_id] = {"status": "failed", "error": str(e)}
    
    background_tasks.add_task(training_task)
    return {"status" : "Isolation Forest training started in background", "task_id" : task_id}

@router.post("/autoencoder", response_model = TrainResponse)
async def train_autoencoder(
    background_tasks: BackgroundTasks, 
    fraud_detector : FraudDetection = Depends(fraud_detection_service.get_fraud_detector)
):
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status":"started"}

    def training_task():
        try:
            result = fraud_detection_service.train_autoencoder(fraud_detector)
            task_statuses[task_id] = {
                "status":"completed",
                "AUPRC":result.get('AUPRC'),
                "y_test":result.get('y_test'),
                "y_pred":result.get('y_pred')
            }
        except Exception as e:
            logger.error(f"Error during autoencoder training: {str(e)}")
            task_statuses[task_id] = {"status": "failed", "error": str(e)}
    
    background_tasks.add_task(training_task)
    return {"status" : "Autoencoder training started in background", "task_id" : task_id}

@router.get("/status/{task_id}", response_model = StatusResponse)
async def get_task_status(task_id: str):
    status = task_statuses.get(task_id, {"status": "not_found"})
    return {
        "task_id": task_id,
        "status": status.get("status", "not found"),
        "AUPRC": status.get("AUPRC"),
        "y_test": status.get("y_test", []),
        "y_pred": status.get("y_pred", [])
    }