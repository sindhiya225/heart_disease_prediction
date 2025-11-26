import subprocess
import time
import sys
import os

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"Command: {command}")
    
    try:
        if sys.platform == "win32":
            # On Windows, use shell=True
            process = subprocess.Popen(command, shell=True)
        else:
            # On Unix-like systems
            process = subprocess.Popen(command, shell=True)
        
        return process
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return None

def main():
    print("🤖 Heart Disease FL - Complete Project Runner")
    print("This script will help you run the entire project step by step")
    
    # Step 1: Create sample data (if needed)
    print("\n1. First, let's check if we have data...")
    if not os.path.exists("data/raw/heart_disease.csv"):
        print("📊 Creating sample data...")
        subprocess.run([sys.executable, "create_sample_data.py"], check=True)
    else:
        print("✅ Data already exists!")
    
    # Step 2: Train centralized model
    print("\n2. Training centralized model...")
    subprocess.run([sys.executable, "centralized/train_centralized.py"], check=True)
    
    # Step 3: Run algorithm comparison
    print("\n3. Running algorithm comparison...")
    subprocess.run([sys.executable, "comparison/algorithm_comparison.py"], check=True)
    
    print("\n🎯 Next steps:")
    print("4. To run Federated Learning:")
    print("   - Open Terminal 1: python federated_learning/server.py")
    print("   - Open Terminal 2: python federated_learning/client.py 0")
    print("   - Open Terminal 3: python federated_learning/client.py 1")
    print("   - Open Terminal 4: python federated_learning/client.py 2")
    print("\n5. To launch Streamlit app:")
    print("   - streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()