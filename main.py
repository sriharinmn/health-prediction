import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
parkinsons_df = pd.read_csv('parkinsons.csv')
heart_df = pd.read_csv('heart.csv')
diabetes_df = pd.read_csv('diabetes.csv')

# Function to preprocess data
def preprocess_data(df, target_column):
    # Drop first column (assumed to contain names or IDs)
    df = df.iloc[:, 1:]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Preprocess each dataset
X_parkinsons, y_parkinsons = preprocess_data(parkinsons_df, 'status')
X_heart, y_heart = preprocess_data(heart_df, 'target')
X_diabetes, y_diabetes = preprocess_data(diabetes_df, 'Outcome')

# Function to train and evaluate a model
def train_and_evaluate(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f'--- {dataset_name} Model Performance ---')
    print(f'Accuracy: {accuracy:.4f}')
    print(report)
    print('\n')

# Train and evaluate models
train_and_evaluate(X_parkinsons, y_parkinsons, 'Parkinsons')
train_and_evaluate(X_heart, y_heart, 'Heart Disease')
train_and_evaluate(X_diabetes, y_diabetes, 'Diabetes')
