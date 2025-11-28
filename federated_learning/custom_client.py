import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.model import HeartDiseaseMLP
from federated_learning.utils import get_client_dataloader
from config import *

class FederatedClient:
    def __init__(self, client_id, server):
        self.client_id = client_id
        self.server = server
        self.local_model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Better optimizer with regularization
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        # Get client data
        self.trainloader = get_client_dataloader(client_id, batch_size=BATCH_SIZE)
        if self.trainloader is None:
            raise ValueError(f"Could not load data for client {client_id}")
        
        print(f"✅ Client {client_id} initialized with {len(self.trainloader.dataset)} samples")
    
    def local_training(self, global_weights, num_epochs=2):  # Increased local epochs
        """Perform local training on client data with better regularization"""
        # Load global weights
        self.local_model.load_state_dict(global_weights)
        self.local_model.train()
        
        print(f"🔧 Client {self.client_id} starting local training...")
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.trainloader)
            print(f"   Client {self.client_id} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Return updated weights and data size
        updated_weights = copy.deepcopy(self.local_model.state_dict())
        data_size = len(self.trainloader.dataset)
        
        return updated_weights, data_size
    
    
    def evaluate_local_model(self):
        """Evaluate local model on client's test data"""
        testloader = get_client_dataloader(self.client_id, batch_size=BATCH_SIZE)
        self.local_model.eval()
        
        correct, total, test_loss = 0, 0, 0.0
        
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.local_model(data)
                loss = self.criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(testloader)
        
        print(f"[Client {self.client_id}] Local Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        # Save metrics
        import os
        os.makedirs("metrics", exist_ok=True)
        with open(f"metrics/client{self.client_id}_metrics.csv", "a") as f:
            f.write(f"{self.server.current_round},{accuracy:.2f},{avg_loss:.4f}\n")
        
        return accuracy, avg_loss
    
    def participate_in_round(self):
        """Participate in one round of federated learning"""
        print(f"🔄 Client {self.client_id} participating in round {self.server.current_round + 1}")
        
        # Get current global weights from server
        global_weights = self.server.get_global_weights()
        
        # Perform local training
        updated_weights, data_size = self.local_training(global_weights, num_epochs=1)
        
        # Send update to server
        self.server.receive_client_update(self.client_id, updated_weights, data_size)
        
        # Evaluate local model
        self.evaluate_local_model()
        
        print(f"✅ Client {self.client_id} completed round {self.server.current_round + 1}")