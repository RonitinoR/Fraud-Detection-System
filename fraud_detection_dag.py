import os
from airflow import DAG
from airflow.operators.python import PythonOperator
import numpy as np
from datetime import datetime
from modeling import FraudDetection
from sklearn.metrics import classification_report

file_path = "transaction_data/creditcard.csv"
temp_dir = "/tmp/airflow_fraud_detection/"

#fucntion to create temporary directory if it doesn't exist
def create_temp_dir():
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

#defining the functions to be used in the DAG
def check_file_path(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    print(f"File {file_path} found.")

def create_fraud_detector(**kwargs):
    print("Creating fraud detection instance.....")
    #initialize the FraudDetection object again in the new task
    fraud_detector = FraudDetection(file_path)
    print("FraudDetection instance has been succesfully created.")
    #push the object to Xcom to use in other tasks
    kwargs['ti'].xcom_push(key = 'status', value = 'fraud_detector_created')
    #save the preprocessed to a temporary file for reuse
    X, y = fraud_detector.load_data()
    X_scaled = fraud_detector.preprocess(X)
    np.savez(os.path.join(temp_dir,'preprocessed_data.npz'), X_scaled = X_scaled, y = y) 

def undersampling_data(**kwargs):
    fraud_detector = FraudDetection(file_path)
    try:
        #load the preprocessed data from the saved temporary file
        data = np.load(os.path.join(temp_dir,'preprocessed_data.npz'))
        X_scaled, y = data['X_scaled'], data['y']
    except FileNotFoundError:
        raise FileNotFoundError("preprocessed data file not found.")
    X_res, y_res = fraud_detector.undersampling(X_scaled, y)
    #save undersampled data to a temporary file
    np.savez(os.path.join(temp_dir,'undersampled_data.npz'), X_res = X_res, y_res = y_res)

def split_data(**kwargs):
    fraud_detector = FraudDetection(file_path)
    try:
        #load the undersampled data
        data = np.load(os.path.join(temp_dir,'undersampled_data.npz'))
        X_res, y_res = data['X_res'], data['y_res']
    except FileNotFoundError:
        raise FileNotFoundError("undersampling data file not found.")
    X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)
    #save the split data to the temporary file
    np.savez(os.path.join(temp_dir, 'split_data.npz'), X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

def isolation_forest(**kwargs):
    fraud_detector = FraudDetection(file_path)
    try:
        #load the split data
        data = np.load(os.path.join(temp_dir,'split_data.npz'))
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    except FileNotFoundError:
        raise FileNotFoundError("Split data file is not found.")
    isolation = fraud_detector.isolationForest(X_train)
    y_pred = fraud_detector.predict(isolation, X_test)
    auc = fraud_detector.evaluate(y_test, y_pred)
    print(f"Isolation forest AUPRC(Area Under Precision-Recall Curve): {auc:.4f}")
    print("\nIsolation forest classification report: ")
    print(classification_report(np.array(y_test), y_pred, zero_division = 1))

def train_autoencoder(**kwargs):
    fraud_detector = FraudDetection(file_path)
    try:
        #load the split data again in here
        data = np.load(os.path.join(temp_dir,'split_data.npz'))
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    except FileNotFoundError:
        raise FileNotFoundError("Split data file is not found.")
    autoencoder = fraud_detector.train_encoder(X_train)
    threshold = fraud_detector.threshold_selection(autoencoder, X_test, y_test)
    print(f"selected balanced threshold with better recall rate: {threshold:.4f}")
    y_pred_ae = fraud_detector.predict_autoencoder(autoencoder, X_test, threshold)
    auc_ae = fraud_detector.evaluate(y_test, y_pred_ae)
    print(f"Autoencoder AUPRC(Area Under Precision_Recall Curve): {auc_ae:.4f}")
    print("\nAutoencoder classification report: ")
    print(classification_report(y_test, y_pred_ae))

def cleanup_temp_files():
    if not os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            os.remove(file_path)
        os.rmdir(temp_dir)
        print(f"Cleaned up temporary files in {temp_dir}")
default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('fraud_detection_pipeline', default_args = default_args, schedule_interval= '@daily', catchup = False) as dag:
    create_temp_dir_task = PythonOperator(
        task_id = 'create_temp_dir',
        python_callable = create_temp_dir
    )
    check_file_task = PythonOperator(
        task_id = 'check_file_path',
        python_callable = check_file_path,
        op_args = [file_path]
    )

    create_fraud_detector_task = PythonOperator(
        task_id = 'create_fraud_detector',
        python_callable = create_fraud_detector
    )

    undersample_task = PythonOperator(
        task_id = 'undersampling_data',
        python_callable = undersampling_data
    )

    split_data_task = PythonOperator(
        task_id = 'split_data',
        python_callable = split_data
    )

    isolation_forest_task = PythonOperator(
        task_id = 'isolation_forest',
        python_callable = isolation_forest
    )

    train_autoencoder_task = PythonOperator(
        task_id = 'train_autoencoder',
        python_callable = train_autoencoder
    )
    cleanup_task = PythonOperator(
        task_id = 'cleanup_temp_files',
        python_callable = cleanup_temp_files
    )
    #Defining the task dependencies
    create_temp_dir_task >> check_file_task >> create_fraud_detector_task >> undersample_task >> split_data_task >> isolation_forest_task >> train_autoencoder_task >> cleanup_task