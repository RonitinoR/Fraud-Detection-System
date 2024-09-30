from fastapi import APIRouter, Depends, HTTPException, Query
import matplotlib.pyplot as plt
import io
import base64
import logging
from app.services import fraud_detection_service
from app.models.schemas import VizualizationResponse
from modeling import FraudDetection
from sklearn.metrics import precision_recall_curve
from typing import Optional

router = APIRouter()

#Initialize logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def plot_precision_recall_curve(y_test, y_pred):
    """Generated a precision recall curve and returns the image in base64 encoded string."""
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        #create plotly figure
        plt.figure(figsize = (10,6))
        plt.plot(precision, recall, marker = '.', label = 'Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid()
        
        #Convert plot to png image
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)
        image_png = buf.getvalue()
        buf.close()
        plt.close()

        #Encode the image to base64
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating precision-recall curve: {e}")
        raise HTTPException(status_code = 500, detail = "Failed to generate precision-recall curve")

def get_model_result(model_type: str, fraud_detector: FraudDetection):
    """Handles model selection and training, returning the result dictionary."""
    try:
        if model_type == 'isolation_forest':
            result = fraud_detection_service.train_isolation_forest(fraud_detector)
        elif model_type == 'autoencoder':
            result = fraud_detection_service.train_autoencoder(fraud_detector)
        else: raise HTTPException(status_code= 400, detail= "Invalid model type specified")

        if not result or 'y_test' not in result or 'y_pred' not in result:
            logger.error(f"Missing expected keys in result for model type: {model_type}")
            raise HTTPException(status_code = 500, detail = "Model training failed to return expected results")
        return result
    except ValueError as ve:
        logger.warning(ve)
        raise HTTPException(status_code = 400, detail = str(ve))
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code = 500, detail = "An unexpected error occurred during model training")

@router.get("/precision_recall", response_model = VizualizationResponse)
async def visualize_precision_recall(
    model_type: str = Query(..., description = "Type of model to train ('isolation_forest' or 'autoencoder')"),
    fraud_detector: FraudDetection = Depends(fraud_detection_service.get_fraud_detector)
):
    """Endpoint to visualize the precision-recall curve for a specified model type"""
    logger.info(f"Starting precision-recall visualization for model: {model_type}")

    #obtain the training result
    result = get_model_result(model_type, fraud_detector)
    
    #generate the graph
    image_data = plot_precision_recall_curve(result['y_test'], result['y_pred'])
    logger.info(f"Precision-recall curve successfully generated for model: {model_type}")
    
    return {'image_data': image_data}
