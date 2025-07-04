#!/usr/bin/env python3
"""
Simple launcher for 0XVENOM Web Interface
This script ensures proper imports and launches the Streamlit app
"""

import os
import sys
import subprocess

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the web interface script
    web_script = os.path.join(current_dir, "scripts", "web_interface.py")
    
    # Check if the web interface file exists
    if not os.path.exists(web_script):
        print("❌ Web interface script not found!")
        return
    
    # Set environment variables for proper imports
    env = os.environ.copy()
    env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    # Launch Streamlit
    try:
        print("🚀 Starting 0XVENOM Web Interface...")
        print("🌐 Open your browser and go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            web_script,
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down 0XVENOM Web Interface...")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

if __name__ == "__main__":
    main()
