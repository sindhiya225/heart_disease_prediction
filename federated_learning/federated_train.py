import threading
import time
import pandas as pd
from custom_server import FederatedServer
from custom_client import FederatedClient
from utils import initialize_client_data
from config import *

def run_federated_learning():
    """Main function to run custom federated learning"""
    print("🤖 Starting Custom Federated Learning (No External Libraries)")
    print("=" * 60)
    
    # Initialize data
    initialize_client_data()
    
    # Create server
    server = FederatedServer(num_clients=NUM_CLIENTS)
    
    # Create clients
    clients = []
    for client_id in range(NUM_CLIENTS):
        try:
            client = FederatedClient(client_id, server)
            clients.append(client)
        except Exception as e:
            print(f"❌ Failed to create client {client_id}: {e}")
    
    if len(clients) < NUM_CLIENTS:
        print(f"❌ Only {len(clients)} clients created. Need {NUM_CLIENTS}.")
        return
    
    # Clear and initialize metrics files with headers
    import os
    os.makedirs("metrics", exist_ok=True)
    
    # Initialize all metrics files with headers
    metrics_files = {
        "global_metrics.csv": "round,accuracy,loss\n",
        "client0_metrics.csv": "round,accuracy,loss\n",
        "client1_metrics.csv": "round,accuracy,loss\n", 
        "client2_metrics.csv": "round,accuracy,loss\n"
    }
    
    for filename, header in metrics_files.items():
        filepath = f"metrics/{filename}"
        with open(filepath, "w") as f:
            f.write(header)
        print(f"✅ Initialized {filepath}")
    
    # Run federated learning rounds
    def client_participation_worker(client, round_num):
        """Worker function for client participation"""
        time.sleep(1)  # Stagger client starts
        client.participate_in_round()
    
    for round_num in range(FED_ROUNDS):
        print(f"\n🎯 Starting Federated Learning Round {round_num + 1}/{FED_ROUNDS}")
        
        # Start all clients in separate threads
        threads = []
        for client in clients:
            thread = threading.Thread(target=client_participation_worker, args=(client, round_num))
            threads.append(thread)
            thread.start()
        
        # Wait for all clients to finish
        for thread in threads:
            thread.join()
        
        # Server aggregates updates
        server.aggregate_updates()
        
        print(f"✅ Round {round_num + 1} completed!")
        print("-" * 50)
    
    print("\n🎊 Custom Federated Learning Completed Successfully!")
    print("📊 Results saved in 'metrics/' directory")
    print("🤖 Models saved in 'models/' directory")
    
    # Verify metrics files
    print("\n🔍 Verifying metrics files...")
    for filename in metrics_files.keys():
        filepath = f"metrics/{filename}"
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"✅ {filename}: {len(df)} records")
            except Exception as e:
                print(f"❌ {filename}: Error - {e}")
        else:
            print(f"❌ {filename}: Not found")

if __name__ == "__main__":
    run_federated_learning()