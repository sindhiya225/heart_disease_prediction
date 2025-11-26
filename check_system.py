import os
import pandas as pd
import torch

def check_system():
    print("🔍 System Diagnostic Check")
    print("=" * 50)
    
    # Check directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    METRICS_DIR = os.path.join(BASE_DIR, "metrics")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Metrics directory: {METRICS_DIR} - {os.path.exists(METRICS_DIR)}")
    print(f"📁 Models directory: {MODELS_DIR} - {os.path.exists(MODELS_DIR)}")
    
    # Check model files
    print("\n🤖 Model Files:")
    model_files = [
        "centralized_model.pth",
        "global_model.pth"
    ]
    
    for model_file in model_files:
        model_path = os.path.join(MODELS_DIR, model_file)
        exists = os.path.exists(model_path)
        size = os.path.getsize(model_path) if exists else 0
        print(f"  {model_file}: {'✅' if exists else '❌'} ({size} bytes)")
        
        if exists and size > 0:
            try:
                # Try to load the model
                if "centralized" in model_file:
                    from federated_learning.model import HeartDiseaseMLP
                    model = HeartDiseaseMLP(13, 64, 2)
                    model.load_state_dict(torch.load(model_path, map_location="cpu"))
                    print(f"    ✅ Model loads successfully")
            except Exception as e:
                print(f"    ❌ Error loading model: {e}")
    
    # Check metrics files
    print("\n📊 Metrics Files:")
    metrics_files = [
        "centralized_metrics.csv",
        "global_metrics.csv",
        "client0_metrics.csv", 
        "client1_metrics.csv",
        "client2_metrics.csv"
    ]
    
    for metrics_file in metrics_files:
        metrics_path = os.path.join(METRICS_DIR, metrics_file)
        exists = os.path.exists(metrics_path)
        if exists:
            size = os.path.getsize(metrics_path)
            try:
                df = pd.read_csv(metrics_path)
                print(f"  {metrics_file}: ✅ ({size} bytes, {len(df)} rows)")
                print(f"    Columns: {df.columns.tolist()}")
                if not df.empty:
                    print(f"    First row: {dict(df.iloc[0])}")
            except Exception as e:
                print(f"  {metrics_file}: ❌ Error reading: {e}")
        else:
            print(f"  {metrics_file}: ❌ Not found")

if __name__ == "__main__":
    check_system()