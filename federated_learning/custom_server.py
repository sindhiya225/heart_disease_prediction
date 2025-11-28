import torch
import threading
import time
import json
import socketserver
import http.server
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.model import HeartDiseaseMLP
from federated_learning.utils import get_test_dataloader
from config import *

class FederatedServer:
    def __init__(self, num_clients=3):
        self.num_clients = num_clients
        self.global_model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.client_weights = {}
        self.client_data_sizes = {}
        self.current_round = 0
        self.server_port = 8080
        
    def federated_averaging(self, client_weights, client_data_sizes):
        """Implement Federated Averaging algorithm from scratch"""
        print("🔄 Performing Federated Averaging...")
        
        # Initialize averaged weights with zeros
        averaged_weights = {}
        for key in client_weights[0].keys():
            averaged_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Calculate total data size
        total_size = sum(client_data_sizes)
        
        # Weighted average based on client data sizes
        for client_idx, weights in enumerate(client_weights):
            client_size = client_data_sizes[client_idx]
            client_weight = client_size / total_size
            
            for key in weights.keys():
                averaged_weights[key] += weights[key] * client_weight
        
        return averaged_weights
    
    def aggregate_updates(self):
        """Aggregate client updates using FedAvg"""
        if len(self.client_weights) < self.num_clients:
            print(f"⚠️ Waiting for more clients. Have {len(self.client_weights)}/{self.num_clients}")
            return False
        
        # Get weights and data sizes from all clients
        client_weights_list = []
        client_data_sizes_list = []
        
        for client_id in range(self.num_clients):
            if client_id in self.client_weights:
                client_weights_list.append(self.client_weights[client_id])
                client_data_sizes_list.append(self.client_data_sizes[client_id])
        
        # Perform federated averaging
        new_global_weights = self.federated_averaging(client_weights_list, client_data_sizes_list)
        
        # Update global model
        self.global_model.load_state_dict(new_global_weights)
        
        # Save global model
        torch.save(self.global_model.state_dict(), "models/global_model.pth")
        
        # Evaluate global model
        self.evaluate_global_model()
        
        # Clear client updates for next round
        self.client_weights.clear()
        self.client_data_sizes.clear()
        
        self.current_round += 1
        return True
    
    def evaluate_global_model(self):
        """Evaluate the global model on test data"""
        testloader = get_test_dataloader(batch_size=BATCH_SIZE)
        self.global_model.eval()
        
        correct, total, test_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in testloader:
                outputs = self.global_model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(testloader)
        
        print(f"🌍 Global Model - Round {self.current_round}: "
              f"Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        # Save metrics
        import os
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/global_metrics.csv", "a") as f:
            f.write(f"{self.current_round},{accuracy:.2f},{avg_loss:.4f}\n")
    
    def receive_client_update(self, client_id, weights, data_size):
        """Receive model updates from a client"""
        print(f"📥 Received update from Client {client_id} with {data_size} samples")
        self.client_weights[client_id] = weights
        self.client_data_sizes[client_id] = data_size
    
    def get_global_weights(self):
        """Get current global model weights"""
        return self.global_model.state_dict()
    
    def run_server(self, num_rounds=5):
        """Run the federated learning server"""
        print("🚀 Starting Custom Federated Learning Server...")
        print(f"📡 Server will run for {num_rounds} rounds with {self.num_clients} clients")
        
        for round in range(num_rounds):
            print(f"\n{'='*50}")
            print(f"🔄 Starting Round {round + 1}/{num_rounds}")
            print(f"{'='*50}")
            
            # Wait for all client updates
            start_time = time.time()
            while len(self.client_weights) < self.num_clients:
                print(f"⏳ Waiting for clients... ({len(self.client_weights)}/{self.num_clients})")
                time.sleep(2)
                
                # Timeout after 30 seconds
                if time.time() - start_time > 30:
                    print("❌ Timeout waiting for clients")
                    break
            
            # Aggregate updates
            if self.aggregate_updates():
                print(f"✅ Round {round + 1} completed successfully!")
            else:
                print(f"❌ Round {round + 1} failed!")
            
            time.sleep(1)  # Brief pause between rounds
        
        print("\n🎉 Federated Learning completed!")
        print(f"📊 Final global model saved as 'models/global_model.pth'")