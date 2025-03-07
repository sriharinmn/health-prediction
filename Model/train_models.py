import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
parkinsons_df = pd.read_csv("../datasets/parkinsons.csv")
heart_df = pd.read_csv("../datasets/heart.csv")
diabetes_df = pd.read_csv("../datasets/diabetes.csv")


# Function to prepare data
def prepare_data(df, target_column):
    df = df.dropna()  # Remove any missing values
    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


# Neural network model function
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train models
datasets = {
    "parkinsons": ("status", parkinsons_df),
    "heart": ("target", heart_df),
    "diabetes": ("Outcome", diabetes_df),
}

for name, (target, df) in datasets.items():
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target)
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    
    # Save model & scaler
    model.save(f"{name}_model.h5")
    joblib.dump(scaler, f"{name}_scaler.pkl")

print("✅ Models trained and saved.")
