#!/usr/bin/env python3
"""
Setup script for Crypto Trading Strategy System
Installs dependencies and prepares the environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 Setting up Crypto Trading Strategy System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install Python requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Failed to install some dependencies. You may need to install them manually.")
    
    # Create necessary directories
    os.makedirs("crypto_cluster_pca/data", exist_ok=True)
    os.makedirs("crypto_cluster_pca/data_cache/binance_daily", exist_ok=True)
    print("✅ Created necessary directories")
    
    # Check if Jupyter is available
    if run_command("jupyter --version", "Checking Jupyter installation"):
        print("💡 You can now run the analysis notebooks in crypto_cluster_pca/research/")
    
    print("")
    print("🎉 Setup complete!")
    print("=" * 30)
    print("")
    print("📋 Next steps:")
    print("1. Fetch historical data:")
    print("   python crypto_cluster_pca/data_pipeline/binance_data.py")
    print("")
    print("2. Verify the pipeline is causal:")
    print("   python crypto_cluster_pca/data_pipeline/test_causality.py")
    print("")
    print("3. Explore Jupyter notebooks:")
    print("   cd crypto_cluster_pca/research")
    print("   jupyter notebook")

if __name__ == "__main__":
    main()