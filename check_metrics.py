import pandas as pd
import os

def check_metrics_files():
    print("🔍 Checking Metrics Files...")
    print("="*50)
    
    metrics_files = [
        "metrics/centralized_metrics.csv",
        "metrics/global_metrics.csv", 
        "metrics/client0_metrics.csv",
        "metrics/client1_metrics.csv",
        "metrics/client2_metrics.csv"
    ]
    
    for file in metrics_files:
        print(f"\n📊 Checking {file}:")
        if os.path.exists(file):
            try:
                # Check file size
                file_size = os.path.getsize(file)
                print(f"   File size: {file_size} bytes")
                
                if file_size == 0:
                    print("   ❌ File is empty!")
                    continue
                
                # Try to read the file
                with open(file, 'r') as f:
                    content = f.read().strip()
                    print(f"   Content preview: {content[:100]}...")
                
                # Try to load as DataFrame
                df = pd.read_csv(file, header=None)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {df.columns.tolist()}")
                print(f"   First few rows:")
                print(df.head())
                
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print("   ❌ File not found!")

if __name__ == "__main__":
    check_metrics_files()