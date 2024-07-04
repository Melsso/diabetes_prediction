import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

class ModelEvaluator:
    def __init__(self, X_test_path, y_test_path, model_path):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path
        self.model = None
        self.feature_names = None  # Store feature names

    def load_test_data(self):
        X_test = pd.read_csv(self.X_test_path)
        y_test = pd.read_csv(self.y_test_path, header=None).values.ravel()
        return X_test, y_test

    def load_model(self):
        self.model = load(self.model_path)

    def evaluate_model(self):
        X_test, y_test = self.load_test_data()
        self.load_model()

        # Ensure feature names are consistent
        X_test = X_test.values  # Convert to NumPy array or list of lists

        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        y_scores = self.model.predict_proba(X_test)[:, 1]  # Predicted probabilities of class 1
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    X_test_path = os.path.join(os.path.dirname(__file__), '../data/processed/X_test.csv')
    y_test_path = os.path.join(os.path.dirname(__file__), '../data/processed/y_test.csv')
    model_path = os.path.join(os.path.dirname(__file__), '../models/logistic_regression_model.pkl')

    evaluator = ModelEvaluator(X_test_path, y_test_path, model_path)
    evaluator.evaluate_model()
