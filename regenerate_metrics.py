import os
import shutil

def regenerate_metrics():
    """Regenerate all metrics and models from scratch"""
    print("🔄 Regenerating all metrics and models...")
    print("="*50)
    
    # Remove existing metrics and models
    folders_to_clear = ["metrics", "models"]
    
    for folder in folders_to_clear:
        if os.path.exists(folder):
            print(f"🗑️  Clearing {folder}/ directory...")
            shutil.rmtree(folder)
        os.makedirs(folder)
        print(f"✅ Created {folder}/ directory")
    
    # Run centralized training
    print("\n🎯 Running Centralized Training...")
    os.system("python centralized/train_centralized.py")
    
    # Run federated training  
    print("\n🎯 Running Federated Learning...")
    os.system("python federated_learning/federated_train.py")
    
    print("\n✅ All metrics and models regenerated!")
    
    # Verify results
    print("\n🔍 Verification:")
    metrics_files = [
        "metrics/centralized_metrics.csv",
        "metrics/global_metrics.csv",
        "metrics/client0_metrics.csv",
        "metrics/client1_metrics.csv", 
        "metrics/client2_metrics.csv",
        "models/centralized_model.pth",
        "models/global_model.pth"
    ]
    
    for file in metrics_files:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - MISSING OR EMPTY")

if __name__ == "__main__":
    regenerate_metrics()