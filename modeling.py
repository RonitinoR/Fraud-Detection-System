import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.engine import data_adapter
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report
from imblearn.under_sampling import RandomUnderSampler

#this is a patching system to temporarily run the tensdorflow without any attribute error
def _is_distributed_dataset(ds):
    return isinstance(ds,data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

class FraudDetection:
    def __init__(self, filepath: str, undersample_ratio = 0.5, test_size = 0.3, contamination_ratio = 0.00172, epoch = 50, batch_size = 32, min_recall = 0.5):
        """
        Initializing the pipeline with the configuration parameters.

        :param filepath: path to the dataset.
        :param undersample_ratio: the ratio for undersampling the majority class.
        :param test_size: proportion of dataset to be included in the test split.
        :param contamination_ratio: contamination parameter for the isolation forest.
        :param epoch: number of epochs for autoencoder training
        :param batch_size: batch size for autoencoder training 
        :param min_recall: hyperparameter tuning the autoencoder to balance out the anomaly detection system
        """
        self.filepath = filepath
        self.undersample_ratio = undersample_ratio
        self.test_size = test_size
        self.contamination_ratio = contamination_ratio
        self.epoch = epoch
        self.batch_size = batch_size
        self.min_recall = min_recall
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.filepath)
        X = df.drop('Class', axis = 1).values #Features
        y = df['Class'].values
        return X, y
    
    def preprocess(self, X: np.ndarray):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def undersampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r_und = RandomUnderSampler(sampling_strategy= self.undersample_ratio, random_state=42)
        X_res, y_res = r_und.fit_resample(X, y)
        return X_res, y_res
    
    def traintest_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= self.test_size, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def isolationForest(self, X_train: np.ndarray):
        iso = IsolationForest(contamination= self.contamination_ratio, random_state=42)
        iso.fit(X_train)
        return iso
    
    def predict(self, model, X_test: np.ndarray) -> np.ndarray:
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        return y_pred
    
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray):
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall, precision)
        return pr_auc
    
    def autoencoder(self, input_dim):
        input_layer = layers.Input(shape = (input_dim,))
        encoded = layers.Dense(32, activation = "relu")(input_layer)
        encoded = layers.Dense(16, activation = "relu")(encoded)
        encoded = layers.Dense(8, activation = "relu")(encoded)

        decoded = layers.Dense(16, activation = "relu")(encoded)
        decoded = layers.Dense(32, activation = "relu")(decoded)
        decoded = layers.Dense(input_dim, activation = "sigmoid")(decoded)

        autoencoder = models.Model(inputs = input_layer, outputs = decoded)
        autoencoder.compile(optimizer = "adam", loss = "mse")
        return autoencoder

    def train_encoder(self, X_train: np.ndarray):
        input_dim = X_train.shape[1]
        autoencoder = self.autoencoder(input_dim)
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
        autoencoder.fit(X_train, X_train, epochs = self.epoch, batch_size = self.batch_size, validation_split = 0.1, verbose = 1, callbacks = [early_stopping])
        return autoencoder
    
    def predict_autoencoder(self, autoencoder, X_test: np.ndarray, threshold: float):
        X_test_pred =  autoencoder.predict(X_test)
        reconstruction_error = np.mean(np.abs(X_test_pred - X_test), axis = 1)
        y_pred_auto = (reconstruction_error > threshold).astype(int)
        return y_pred_auto
    
    def threshold_selection(self, autoencoder, X_test: np.ndarray, y_test:np.ndarray) -> float:
        X_test_pred = autoencoder.predict(X_test)
        reconstruction_error = np.mean(np.abs(X_test_pred - X_test), axis = 1)
        precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_error)
        
        #get all the threshold values for the class 0 (normal) data having recall rates higher than than the minimum recall rate
        for threshold in thresholds:
            y_pred = (reconstruction_error > threshold).astype(int)
            report = classification_report(y_test, y_pred, output_dict = True)
            if report['0']['recall'] >= self.min_recall:
                return threshold

        #returns the maximum default threshold if none are found
        return thresholds[-1]