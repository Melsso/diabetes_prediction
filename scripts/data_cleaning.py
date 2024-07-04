import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/raw/diabetes.csv')
data = pd.read_csv(data_path)

# Basic information about the dataset
print("\nBasic information about the dataset:")
print(data.info())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Replace zero values in specific columns with NaN
columns_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
data[columns_with_zero_values] = data[columns_with_zero_values].replace(0, pd.NA)

# Impute missing values with the median of each column
result = data.fillna(data.median()).infer_objects(copy=False)

# Standardize the features
scaler = StandardScaler()
columns_to_scale = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
result[columns_to_scale] = scaler.fit_transform(result[columns_to_scale])

# Set feature names
result.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Separate features and target
X = result.drop('Outcome', axis=1)
y = result['Outcome']

# Perform SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Print column names to verify consistency
print("Training data columns:", X_train.columns.tolist())
print("Test data columns:", X_test.columns.tolist())

# Save training and testing sets
X_train.to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_train.csv'), index=False)
X_test.to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/X_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_train.csv'), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(os.path.dirname(__file__), '../data/processed/y_test.csv'), index=False, header=False)
