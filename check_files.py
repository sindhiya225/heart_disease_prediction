import os

def check_file_locations():
    print("🔍 Checking File Locations...")
    print("="*50)
    
    files_to_check = [
        "metrics/centralized_metrics.csv",
        "metrics/global_metrics.csv", 
        "metrics/client0_metrics.csv",
        "metrics/client1_metrics.csv",
        "metrics/client2_metrics.csv",
        "models/centralized_model.pth",
        "models/global_model.pth"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND")
    
    print("\n📁 Current working directory:", os.getcwd())
    print("📁 Files in current directory:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  📂 {item}/")
        else:
            print(f"  📄 {item}")

if __name__ == "__main__":
    check_file_locations()