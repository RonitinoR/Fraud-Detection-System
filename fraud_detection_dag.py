import os
import pandas as pd
import numpy as np
import logging 
from airflow import DAG
from airflow.decorators import task_group, task
from datetime import datetime
from modeling import FraudDetection
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

#define the file path for the dataset
file_path = "transaction_data/creditcard.csv"

#task group for preprocessing
@task_group(group_id = 'preprocess_data')
def data_preprocess(data_path):

    #task to check the existense of dataset
    @task
    def check_file_path(source_file):
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Error: The file {source_file} does not exist.")
        logger.info(f"The file {source_file} found.")
        return source_file
    
    #task to preprocess the data and save the preprocessed data
    @task
    def preprocess_data(source_file):
        fraud_detector = FraudDetection(source_file)
        X, y = fraud_detector.load_data()
        X_scaled = fraud_detector.preprocess(X)
        dest_file = "/tmp/preprocessed_file.npz"
        np.savez(dest_file, X_scaled = X_scaled, y = y)
        logger.info(f"Preprocessed data saved to {dest_file}.")
        return dest_file
    
    #sequential execution of task within the task group
    checked_file = check_file_path(data_path)
    preprocessed_file = preprocess_data(source_file = checked_file)
    
    return preprocessed_file

#task group for model training and evaluating
@task_group(group_id = 'model_training')
def model_training(preprocessed_file):

    #The task for undersampling the data
    @task
    def undersampling_data(source_file):
        fraud_detector = FraudDetection(file_path)
        data = np.load(source_file)
        X_scaled, y = data['X_scaled'], data['y']
        X_res, y_res = fraud_detector.undersampling(X_scaled, y)
        dest_file = "/tmp/undersampled_data.npz"
        np.savez(dest_file, X_res = X_res, y_res = y_res)
        logger.info(f"The undersampled data is saved to {dest_file}.")
        return dest_file
    
    #Task for splitting the data into train and test
    @task
    def splitting(source_file):
        fraud_detector = FraudDetection(file_path)
        data = np.load(source_file)
        X_res, y_res = data['X_res'], data['y_res']
        X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)
        dest_file = "/tmp/split_data.npz"
        np.savez(dest_file, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
        logger.info(f"The split data is saved into {dest_file}.")
        return dest_file
    
    #the task of training and evaluating isolation forest
    @task
    def isolation_forest(source_file):
        fraud_detector = FraudDetection(file_path)
        data = np.load(source_file)
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        isolation = fraud_detector.isolationForest(X_train)
        y_pred = fraud_detector.predict(isolation, X_test)
        auc = fraud_detector.evaluate(y_test, y_pred)
        logger.info(f"The AUPRC for Isolation Forest: {auc:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division = 1)}")

    #the task for training and evaluating autoencoder
    @task
    def train_autoencoder(source_file):
        fraud_detector = FraudDetection(file_path)
        data = np.load(source_file)
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        autoencoder = fraud_detector.train_encoder(X_train)
        threshold = fraud_detector.threshold_selection(autoencoder, X_test, y_test)
        logger.info(f"Selected threshold value for balanced recall rates: {threshold:.4f}")
        y_pred_ae = fraud_detector.predict_autoencoder(autoencoder, X_test, threshold)
        auc_ae = fraud_detector.evaluate(y_test, y_pred_ae)
        logger.info(f"The AUPRC for the Autoencoder: {auc_ae:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred_ae)}")

    #Sequential execution of the task
    undersampled_file = undersampling_data(preprocessed_file)
    split_data = splitting(undersampled_file)
    isolation_forest(split_data)
    train_autoencoder(split_data)

#defining the DAG
default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('fraud_detection_pipeline', default_args = default_args, schedule_interval='@daily', catchup = False) as dag:

    #task group for preprocessing data
    preprocessed_data = data_preprocess(data_path = file_path)

    #task group for model training and evaluation
    model_training(preprocessed_data)