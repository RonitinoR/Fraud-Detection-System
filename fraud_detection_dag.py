import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from modeling import FraudDetection
from sklearn.metrics import classification_report

file_path = "transaction_data/creditcard.csv"

#defining the functions to be used in the DAG
def check_file_path(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
    print(f"File {file_path} found.")

def load_data(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids ='create_fraud_detector')
    X, y = fraud_detector.load_data()
    return X, y

def create_fraud_detector(**kwargs):
    fraud_detector = FraudDetection(file_path)
    #push the object to Xcom to use in other tasks
    kwargs['ti'].xcom_push(key = 'fraud_detector', value = fraud_detector)

def preprocess_data(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids = 'create_fraud_detector')
    X, y = fraud_detector.load_data()
    X_scaled = fraud_detector.preprocess(X)
    return X_scaled, y

def undersampling_data(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids = 'create_fraud_detector')
    X_scaled, y = kwargs['ti'].xcom_pull(task_ids = 'preprocess_data')
    X_res, y_res = fraud_detector.undersampling(X_scaled, y)
    return X_res, y_res

def split_data(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids = 'create_fraud_detector')
    X_res, y_res = kwargs['ti'].xcom_pull(task_ids = 'undersampling_data')
    X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)
    return X_train, X_test, y_train, y_test

def isolation_forest(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids = 'create_fraud_detector')
    X_train, X_test, y_train, y_test = kwargs['ti'].xcom_pull(task_ids = 'split_data')
    isolation = fraud_detector.isolationForest(X_train)
    y_pred = fraud_detector.predict(isolation, X_test)
    auc = fraud_detector.evaluate(y_test, y_pred)
    print(f"Isolation forest AUPRC(Area Under Precision-Recall Curve): {auc:.4f}")
    print("\nIsolation forest classification report: ")
    print(classification_report(y_test, y_pred, zero_division = 1))

def train_autoencoder(**kwargs):
    fraud_detector = kwargs['ti'].xcom_pull(task_ids = 'create_fraud_detector')
    X_train, X_test, y_train, y_test = kwargs['ti'].xcom_pull(task_ids = 'split_data')
    autoencoder = fraud_detector.train_encoder(X_train)
    threshold = fraud_detector.threshold_selection(autoencoder, X_test, y_test)
    print(f"selected balanced threshold with better recall rate: {threshold:.4f}")
    y_pred_ae = fraud_detector.predict_autoencoder(autoencoder, X_test, threshold)
    auc_ae = fraud_detector.evaluate(y_test, y_pred_ae)
    print(f"Autoencoder AUPRC(Area Under Precision_Recall Curve): {auc_ae:.4f}")
    print("\nAutoencoder classification report: ")
    print(classification_report(y_test, y_pred_ae))


default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('fraud_detection_pipeline', default_args = default_args, schedule_interval= '@daily', catchup = False) as dag:
    check_file_task = PythonOperator(
        task_id = 'check_file_path',
        python_callable = check_file_path,
        op_args = [file_path]
    )

    create_fraud_detector_task = PythonOperator(
        task_id = 'create_fraud_detector',
        python_callable = create_fraud_detector,
        provide_context = True
    )

    preprocess_task = PythonOperator(
        task_id = 'preprocess_data',
        python_callable = preprocess_data,
        provide_context = True
    )

    undersample_task = PythonOperator(
        task_id = 'undersampling_data',
        python_callable = undersampling_data,
        provide_context = True
    )

    split_data_task = PythonOperator(
        task_id = 'split_data',
        python_callable = split_data,
        provide_context = True
    )

    isolation_forest_task = PythonOperator(
        task_id = 'isolation_forest',
        python_callable = isolation_forest,
        provide_context = True
    )

    train_autoencoder_task = PythonOperator(
        task_id = 'train_autoencoder',
        python_callable = train_autoencoder,
        provide_context = True
    )

    #Defining the task dependencies
    check_file_task >> create_fraud_detector_task >> preprocess_task >> undersample_task >> split_data_task >> isolation_forest_task >> train_autoencoder_task