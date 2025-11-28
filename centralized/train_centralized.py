import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.model import HeartDiseaseMLP
from federated_learning.utils import load_and_preprocess_data, HeartDiseaseDataset, initialize_client_data
from torch.utils.data import DataLoader
from config import *

def train_centralized():
    """Train model in centralized manner for comparison"""
    print("🔹 Starting Centralized Training...")
    
    # Get absolute paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    METRICS_DIR = os.path.join(BASE_DIR, "metrics")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Initialize client data first (this will create the splits)
    initialize_client_data()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Create datasets and dataloaders
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    test_dataset = HeartDiseaseDataset(X_test, y_test)
    
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Better optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # Clear and initialize metrics file with header
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_file = os.path.join(METRICS_DIR, "centralized_metrics.csv")
    
    # Write header (overwrite if file exists)
    with open(metrics_file, "w") as f:
        f.write("epoch,accuracy,loss\n")
    print(f"✅ Initialized metrics file: {metrics_file}")
    
    # Training loop
    print("🚀 Starting training...")
    best_accuracy = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        avg_train_loss = running_loss / len(trainloader)
        
        # Evaluate
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(testloader)
        
        print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%")
        
        # Save metrics (append mode)
        with open(metrics_file, "a") as f:
            f.write(f"{epoch+1},{accuracy:.2f},{avg_test_loss:.4f}\n")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(MODELS_DIR, "centralized_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model with accuracy: {accuracy:.2f}%")

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f"🎯 Final Model Accuracy: {final_accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    train_centralized()