import os
import sys

def test_paths():
    """Test if we can access the correct directories"""
    print("🧪 Testing Path Access")
    print("=" * 40)
    
    # Current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we can access the main directories
    directories = ["metrics", "models", "data", "centralized", "federated_learning"]
    
    for dir_name in directories:
        dir_path = os.path.join(cwd, dir_name)
        exists = os.path.exists(dir_path)
        print(f"📁 {dir_name}: {'✅ EXISTS' if exists else '❌ MISSING'}")
        if exists:
            try:
                files = os.listdir(dir_path)
                print(f"   Files: {files[:5]}{'...' if len(files) > 5 else ''}")
            except Exception as e:
                print(f"   Error listing: {e}")
    
    # Test writing a file
    print("\n📝 Testing file writing...")
    test_file = os.path.join(cwd, "metrics", "test_file.txt")
    try:
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "w") as f:
            f.write("Test content")
        print(f"✅ Successfully wrote to: {test_file}")
        os.remove(test_file)  # Clean up
    except Exception as e:
        print(f"❌ Failed to write file: {e}")

if __name__ == "__main__":
    test_paths()