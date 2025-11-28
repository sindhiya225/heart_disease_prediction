import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_SIZE

class HeartDiseaseMLP(nn.Module):
    def __init__(self, input_size=None, hidden_size=64, output_size=2):
        super(HeartDiseaseMLP, self).__init__()
        # Use provided input_size or default from config
        self.input_size = input_size if input_size is not None else INPUT_SIZE
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CustomFedAvgAlgorithm:
    """Custom implementation of Federated Averaging algorithm"""
    
    @staticmethod
    def weighted_average(models, data_sizes):
        """
        Implement Federated Averaging from scratch
        models: list of model state dictionaries
        data_sizes: list of data sizes for each client
        """
        total_data = sum(data_sizes)
        averaged_weights = {}
        
        # Initialize with zeros
        for key in models[0].keys():
            averaged_weights[key] = torch.zeros_like(models[0][key])
        
        # Weighted average
        for model, data_size in zip(models, data_sizes):
            weight = data_size / total_data
            for key in model.keys():
                averaged_weights[key] += model[key] * weight
        
        return averaged_weights
    
    @staticmethod
    def client_update(model, dataloader, epochs=1, lr=0.001):
        """Client-side local SGD update"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(epochs):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()