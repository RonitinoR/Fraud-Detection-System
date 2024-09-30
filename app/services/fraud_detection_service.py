import numpy as np
import hashlib
import logging
from fastapi import HTTPException
from cachetools import TTLCache
from modeling import FraudDetection
from app.core.config import settings

#initialize the logger
logger = logging.getLogger(__name__)

#cache to store the recent model results
cache = TTLCache(maxsize=10, ttl = settings.model_cache_expiry)

#utility function to create hash of the dataset for caching purposes
def get_hash_data(data):
    return hashlib.md5(data.tobytes()).hexdigest()

#utility function for caching model results
def cache_model_result(key, result):
    cache[key]= result

#initialize the fraud detection instance
def get_fraud_detector():
    return FraudDetection(filepath=settings.file_path)

def train_isolation_forest(fraud_detector: FraudDetection):
    try:
        logger.info("Starting isolation forest training....")

        #load and hash the data for caching
        X, y = fraud_detector.load_data()
        data_hash = get_hash_data(X)
        cache_key = f'isolation_forest_{data_hash}'

        if cache_key in cache:
            logger.info("Returning cached result for Isolation forest.")
            return cache[cache_key]
        
        #preprocess the data
        X_scaled = fraud_detector.preprocess(X)
        X_res, y_res = fraud_detector.undersampling(X_scaled, y)
        X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)

        #train isolation forest
        isolation_model = fraud_detector.isolationForest(X_train)
        y_pred = fraud_detector.predict(isolation_model, X_test)
        pr_auc = fraud_detector.evaluate(y_test, y_pred)

        result = {
            "AUPRC" : pr_auc,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist()
        }
        cache_model_result(cache_key, result)
        logger.info("Isolation forest training successfully completed.")
        return result
    
    except Exception as e:
        logger.error(f"Error during isolation forest: {str(e)}")
        raise HTTPException(status_code = 500, detail = f"Error during isolation forest: {str(e)}")

def train_autoencoder(fraud_detector: FraudDetection):
    try:
        logger.info("Starting autoencoder training....")
        
        #load and hash the data
        X, y = fraud_detector.load_data()
        data_hash = get_hash_data(X)
        cache_key = f'autoencoder_{data_hash}'
        
        if cache_key in cache:
            logger.info("Returning cached info for autoencoder.")
            return cache[cache_key]
        
        #preprocess the data
        X_scaled = fraud_detector.preprocess(X)
        X_res, y_res = fraud_detector.undersampling(X_scaled,y)
        X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)

        #train autoencoder
        autoencoder = fraud_detector.train_encoder(X_train)
        threshold = fraud_detector.threshold_selection(autoencoder, X_test, y_test)
        y_pred_ae = fraud_detector.predict_autoencoder(autoencoder, X_test, threshold)
        pr_auc = fraud_detector.evaluate(y_test, y_pred_ae)

        result = {
            "AUPRC" : pr_auc,
            "y_test": y_test.tolist(),
            "y_pred": y_pred_ae.tolist()
        }

        #cache the result
        cache_model_result(cache_key, result)
        logger.info("Autoencoder training successfully completed.")
        return result
    except Exception as e:
        logger.error(f"Error during autoencoder training: {str(e)}")
        raise HTTPException(status_code= 500, detail= f"Error during autoencoder training: {str(e)}")
