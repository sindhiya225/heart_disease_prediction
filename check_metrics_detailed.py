# check_metrics_detailed.py
import pandas as pd
import os

def check_metrics_detailed():
    print("🔍 Detailed Metrics Analysis")
    print("="*50)
    
    metrics_files = {
        "centralized": "metrics/centralized_metrics.csv",
        "global": "metrics/global_metrics.csv",
        "client0": "metrics/client0_metrics.csv", 
        "client1": "metrics/client1_metrics.csv",
        "client2": "metrics/client2_metrics.csv"
    }
    
    for name, filepath in metrics_files.items():
        print(f"\n📊 {name.upper()} METRICS:")
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"   File exists, size: {file_size} bytes")
            
            if file_size > 0:
                try:
                    # Read raw content
                    with open(filepath, 'r') as f:
                        content = f.read().strip()
                        print(f"   Raw content: '{content}'")
                    
                    # Try to load as DataFrame
                    df = pd.read_csv(filepath)
                    print(f"   DataFrame shape: {df.shape}")
                    print(f"   Columns: {df.columns.tolist()}")
                    if not df.empty:
                        print(f"   First 3 rows:")
                        print(df.head(3))
                    else:
                        print("   DataFrame is empty!")
                        
                except Exception as e:
                    print(f"   ❌ Error reading: {e}")
            else:
                print("   ❌ File is empty!")
        else:
            print("   ❌ File not found!")

if __name__ == "__main__":
    check_metrics_detailed()