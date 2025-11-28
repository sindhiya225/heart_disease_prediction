import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.utils import load_and_preprocess_data
from federated_learning.model import HeartDiseaseMLP
from config import *

def compare_algorithms():
    """Compare multiple machine learning algorithms"""
    print("🔄 Comparing Machine Learning Algorithms...")
    
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Traditional ML models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }
    
    results = {}
    
    # Train and evaluate traditional models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'model': model,
            'predictions': y_pred
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Evaluate neural network models
    nn_models = {}
    
    # Centralized NN
    try:
        centralized_nn = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        centralized_nn.load_state_dict(torch.load("../models/centralized_model.pth"))
        centralized_nn.eval()
        nn_models['Centralized NN'] = centralized_nn
    except:
        print("Centralized NN model not found")
    
    # Federated NN
    try:
        federated_nn = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        federated_nn.load_state_dict(torch.load("../models/global_model.pth"))
        federated_nn.eval()
        nn_models['Federated NN'] = federated_nn
    except:
        print("Federated NN model not found")
    
    # Evaluate neural networks
    for name, model in nn_models.items():
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            outputs = model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predictions.numpy())
            results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': predictions.numpy()
            }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save comparison results
    comparison_df = pd.DataFrame([
        {'Algorithm': algo, 'Accuracy': result['accuracy']}
        for algo, result in results.items()
    ])
    
    os.makedirs("../comparison/results", exist_ok=True)
    comparison_df.to_csv("../comparison/results/algorithm_comparison.csv", index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    accuracies = [results[algo]['accuracy'] for algo in algorithms]
    
    bars = plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet'])
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Heart Disease Prediction: Algorithm Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../comparison/results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Algorithm comparison completed!")
    return results

if __name__ == "__main__":
    compare_algorithms()