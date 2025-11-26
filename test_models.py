# test_models.py - Run this to check your models
import os
import torch
from federated_learning.model import HeartDiseaseMLP
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

def test_model_loading():
    print("🧪 Testing Model Loading...")
    
    # Check directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"📁 Exists: {os.path.exists(MODELS_DIR)}")
    
    if os.path.exists(MODELS_DIR):
        print(f"📄 Files in models/: {os.listdir(MODELS_DIR)}")
    
    # Test creating and loading a model
    print("\n🔧 Testing model creation...")
    model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    print(f"✅ Model created: {model}")
    
    # Test prediction
    print("\n🎯 Testing prediction...")
    test_input = torch.randn(1, INPUT_SIZE)
    output = model(test_input)
    print(f"✅ Prediction works! Output shape: {output.shape}")
    
    # Save a test model
    print("\n💾 Saving test model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    test_path = os.path.join(MODELS_DIR, "test_model.pth")
    torch.save(model.state_dict(), test_path)
    print(f"✅ Test model saved: {test_path}")
    
    # Load the test model
    print("\n📥 Loading test model...")
    loaded_model = HeartDiseaseMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    loaded_model.load_state_dict(torch.load(test_path))
    loaded_model.eval()
    print("✅ Test model loaded successfully!")
    
    # Test loaded model
    output_loaded = loaded_model(test_input)
    print(f"✅ Loaded model prediction works! Output shape: {output_loaded.shape}")

if __name__ == "__main__":
    test_model_loading()