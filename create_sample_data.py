import pandas as pd
import numpy as np
import os
from config import *

def create_sample_heart_disease_data():
    """Create a sample heart disease dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic heart disease data with realistic features
    age = np.random.normal(55, 10, n_samples).astype(int)
    sex = np.random.randint(0, 2, n_samples)  # 0: Female, 1: Male
    cp = np.random.randint(0, 4, n_samples)   # Chest pain type
    trestbps = np.random.normal(130, 20, n_samples).astype(int)  # Resting BP
    chol = np.random.normal(250, 50, n_samples).astype(int)      # Cholesterol
    fbs = np.random.randint(0, 2, n_samples)  # Fasting blood sugar
    restecg = np.random.randint(0, 3, n_samples)  # Resting ECG
    thalach = np.random.normal(150, 20, n_samples).astype(int)  # Max heart rate
    exang = np.random.randint(0, 2, n_samples)  # Exercise induced angina
    oldpeak = np.round(np.random.exponential(1, n_samples), 1)  # ST depression
    slope = np.random.randint(0, 3, n_samples)  # Slope of peak exercise ST
    ca = np.random.randint(0, 4, n_samples)     # Number of major vessels
    thal = np.random.randint(0, 3, n_samples)   # Thalassemia
    
    # Create features matrix
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on some relationship (simulating heart disease risk)
    risk_score = (
        (age - 50) * 0.1 + 
        sex * 0.3 + 
        cp * 0.2 + 
        (trestbps - 120) * 0.01 + 
        (chol - 200) * 0.002 +
        exang * 0.4 +
        oldpeak * 0.3
    )
    
    # Add some noise and create binary target
    df['target'] = (risk_score + np.random.normal(0, 0.5, n_samples) > 0.5).astype(int)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Created sample heart disease dataset at: {RAW_DATA_PATH}")
    print(f"✅ Dataset shape: {df.shape}")
    print(f"✅ Target distribution:\n{df['target'].value_counts()}")
    print(f"✅ Features: {list(df.columns[:-1])}")
    
    return df

if __name__ == "__main__":
    create_sample_heart_disease_data()