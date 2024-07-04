import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

class DiabetesPredictor:
    def __init__(self, X_train_path, y_train_path, model_save_path):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.model_save_path = model_save_path
        self.model = None
        self.feature_names = None  # Store feature names

    def load_data(self):
        data = pd.read_csv(self.X_train_path)
        self.feature_names = data.columns.tolist()  # Store feature names
        X_train = data.values
        y_train = pd.read_csv(self.y_train_path, header=None).values.ravel()
        return X_train, y_train

    def train_model(self, X_train, y_train):
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

    def save_model(self):
        dump(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

if __name__ == "__main__":
    X_train_path = os.path.join(os.path.dirname(__file__), '../data/processed/X_train.csv')
    y_train_path = os.path.join(os.path.dirname(__file__), '../data/processed/y_train.csv')
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/logistic_regression_model.pkl')

    predictor = DiabetesPredictor(X_train_path, y_train_path, model_save_path)
    X_train, y_train = predictor.load_data()

    # Print loaded feature names to ensure consistency
    print("Feature names in loaded data:", predictor.feature_names)

    predictor.train_model(X_train, y_train)
    predictor.save_model()
