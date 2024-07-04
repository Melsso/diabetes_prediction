# Diabetes Prediction Project

This project aims to predict whether a patient has diabetes using logistic regression.

## Project Structure

- `data/`: Contains the dataset files.
  - `raw/`: Original, unprocessed dataset.
  - `processed/`: Cleaned and preprocessed dataset.
- `notebooks/`: Jupyter notebooks for EDA, data cleaning, and model training.
- `scripts/`: Python scripts for data cleaning, model training, and evaluation.
- `models/`: Saved models.
- `reports/`: Reports and documentation.
- `requirements.txt`: Dependencies and libraries required for the project.
- `README.md`: Project overview and setup instructions.

## Setup Instructions

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/diabetes_prediction_project.git
   cd diabetes_prediction_project
```

2. Create and activate a virtual environment:
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
    pip install -r requirements.txt
```

## Usage

1. Run the Jupyter notebooks in the notebooks/ directory to perform EDA, data cleaning, and model training.

2. Use the scripts in the scripts/ directory for data cleaning, model training, and evaluation:
```bash
    python scripts/data_cleaning.py
    python scripts/train_model.py
    python scripts/evaluate_model.py
```

## Results
The results of the project, including the trained model and performance metrics, are stored in the models/ and reports/ directories.
