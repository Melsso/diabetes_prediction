import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/raw/diabetes.csv')
data = pd.read_csv(data_path)

# Impute missing values with the median of each column
result = data.fillna(data.median()).infer_objects(copy=False)

# Standardize the features
# scaler = StandardScaler()
# columns_to_scale = ['Pregnancies', 'Age', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
# result[columns_to_scale] = scaler.fit_transform(result[columns_to_scale])

# Set feature names
result.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Separate the features and the target
X = result.drop('Outcome', axis=1)
y = result['Outcome']

# # Perform SMOTE oversampling
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print column names to verify consistency
print("Training data columns:", X_train.columns.tolist())
print("Test data columns:", X_test.columns.tolist())

if X_train.columns.tolist() != X_test.columns.tolist():
    print('Error: Column names are not the same!')
    os.exit()
# Save training and testing sets
X_train.to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_train.csv'), index=False)
X_test.to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_train.csv'), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_test.csv'), index=False, header=False)
