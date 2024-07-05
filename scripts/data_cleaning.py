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

mean_col = ['Glucose', 'BloodPressure']
med_col = ['SkinThickness', 'Insulin', 'BMI']

dfc = data.copy(deep=True)

for item in mean_col:
    dfc[item].fillna(dfc[item].mean(), inplace=False)
for item in med_col:
    dfc[item].fillna(dfc[item].median(), inplace=False)

dfc.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# Standardize the features
scaler = StandardScaler()
columns_to_scale = ['Pregnancies', 'Age', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
dfc[columns_to_scale] = scaler.fit_transform(dfc[columns_to_scale])

# Set feature names

# Separate the features and the target
X = dfc.drop('Outcome', axis=1)
y = data['Outcome']

# Split resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

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
