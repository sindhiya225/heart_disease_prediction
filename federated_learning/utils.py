import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class HeartDiseaseDataset(Dataset):
    def __init__(self, features, labels):
        # Convert to numpy arrays if they are pandas Series/DataFrame
        if hasattr(features, 'values'):
            features = features.values
        if hasattr(labels, 'values'):
            labels = labels.values
            
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data():
    """Load and preprocess heart disease data"""
    try:
        # Load data
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Loaded dataset with shape: {df.shape}")
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Check if 'target' column exists
        if 'target' not in df.columns:
            print("❌ 'target' column not found in dataset. Please check your CSV file.")
            # Try to find alternative target columns
            possible_targets = ['target', 'class', 'label', 'heart_disease', 'disease', 'HeartDisease']
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col:
                df = df.rename(columns={target_col: 'target'})
                print(f"✅ Renamed '{target_col}' column to 'target'")
            else:
                # If no target column found, use the last column as target
                target_col = df.columns[-1]
                df = df.rename(columns={target_col: 'target'})
                print(f"⚠️ Using last column '{target_col}' as target")
        
        # Encode categorical variables if any
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
            print(f"✅ Encoded categorical column: {col}")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"✅ Target distribution:\n{y.value_counts()}")
        
        # Update input size in config based on actual data
        global INPUT_SIZE
        INPUT_SIZE = X.shape[1]
        print(f"✅ Updated INPUT_SIZE to: {INPUT_SIZE}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Create sample data for testing
        print("🔄 Creating sample data for testing...")
        return create_sample_data()

def create_sample_data():
    """Create sample heart disease data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic heart disease data
    age = np.random.normal(55, 10, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.normal(130, 20, n_samples)
    chol = np.random.normal(250, 50, n_samples)
    fbs = np.random.randint(0, 2, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.normal(150, 20, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.random.exponential(1, n_samples)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.randint(0, 3, n_samples)
    
    # Create features matrix
    X = np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    
    # Generate target based on some relationship
    risk_score = (age - 50) * 0.1 + sex * 0.3 + cp * 0.2 + (trestbps - 120) * 0.01 + (chol - 200) * 0.002
    y = (risk_score + np.random.normal(0, 0.5, n_samples) > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Update input size
    global INPUT_SIZE
    INPUT_SIZE = X.shape[1]
    
    print("✅ Created sample heart disease data for testing")
    print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"✅ Target distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    print(f"✅ INPUT_SIZE set to: {INPUT_SIZE}")
    
    return X_train, X_test, y_train, y_test, scaler

def split_data_for_clients(X_train, y_train, num_clients=3):
    """Split data for federated learning clients"""
    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
    
    # Convert to numpy arrays if they are pandas Series
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Split indices for each client
    indices = np.random.permutation(len(X_train))
    split_indices = np.array_split(indices, num_clients)
    
    client_data = {}
    for i, client_indices in enumerate(split_indices):
        client_X = X_train[client_indices]
        client_y = y_train[client_indices]
        
        # Save client data
        client_df = pd.DataFrame(client_X)
        client_df['target'] = client_y
        client_df.to_csv(os.path.join(SPLIT_DATA_DIR, f'client{i}.csv'), index=False)
        
        client_data[f'client{i}'] = (client_X, client_y)
        
        print(f"✅ Created client {i} data with {len(client_X)} samples")
    
    return client_data

def get_client_dataloader(client_id, batch_size=32):
    """Get DataLoader for a specific client"""
    client_path = os.path.join(SPLIT_DATA_DIR, f'client{client_id}.csv')
    
    try:
        client_df = pd.read_csv(client_path)
        
        X = client_df.drop('target', axis=1).values
        y = client_df['target'].values
        
        dataset = HeartDiseaseDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"❌ Error loading client {client_id} data: {e}")
        # Return empty dataloader or handle appropriately
        return None

def get_test_dataloader(batch_size=32):
    """Get DataLoader for test set"""
    _, X_test, _, y_test, _ = load_and_preprocess_data()
    dataset = HeartDiseaseDataset(X_test, y_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def initialize_client_data():
    """Initialize client data splits"""
    print("🔄 Initializing client data splits...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    split_data_for_clients(X_train, y_train, NUM_CLIENTS)
    print("✅ Client data initialization completed!")