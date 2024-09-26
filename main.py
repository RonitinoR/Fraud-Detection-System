import os
from modeling import FraudDetection
from sklearn.metrics import classification_report
def run(file_path: str):

    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return
    
    #creating an instance for the fraud detection system
    fraud_detector = FraudDetection(file_path)

    #load the data
    X, y = fraud_detector.load_data()

    #preprocessing the data for standard scaling
    X_scaled = fraud_detector.preprocess(X)

    #Undersampling the data
    X_res, y_res = fraud_detector.undersampling(X_scaled, y)

    #Split the data into train and test
    X_train, X_test, y_train, y_test = fraud_detector.traintest_split(X_res, y_res)

    #Train the model
    isolation = fraud_detector.isolationForest(X_train)

    #predict and evaluate the model on Isolation Forest
    y_pred = fraud_detector.predict(isolation, X_test)
    Auc = fraud_detector.evaluate(y_test, y_pred) 
    print(f"Isolation Forest AUPRC: {Auc:.4f}")

    #classification report of the model
    print("\nIsolation forest classification report:")
    print(classification_report(y_test, y_pred, zero_division = 1))

    #training autoencoder
    autoencoder = fraud_detector.train_encoder(X_train)

    #choosing the threshold that provides better recall rate
    threshold = fraud_detector.threshold_selection(autoencoder, X_test, y_test)
    print(f"selected threshold for 95% recall rate: {threshold:.4f}")

    #predict and evaluate the autoencoder
    y_pred_ae = fraud_detector.predict_autoencoder(autoencoder, X_test, threshold)
    auc_ae = fraud_detector.evaluate(y_test, y_pred_ae)
    print(f"Autoencoder AUPRC: {auc_ae:.4f}")

    print("\nClassification report for autoencoder:")
    print(classification_report(y_test, y_pred_ae))

if __name__ == "__main__":
    file_path = 'transaction_data/creditcard.csv'

    run(file_path)