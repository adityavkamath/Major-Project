#!/usr/bin/env python3
"""
Startup script for the Handwritten Equation Solver
Runs both FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import sys
import os
import signal
import threading

def run_fastapi():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ Error running FastAPI server: {e}")

def run_streamlit():
    """Run the Streamlit frontend server"""
    print("🎨 Starting Streamlit frontend server...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit server: {e}")

def main():
    print("🧮 Handwritten Equation Solver")
    print("=" * 50)
    
    # Try to load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env file")
    except ImportError:
        print("ℹ️  python-dotenv not installed, trying to install...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=True)
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ python-dotenv installed and .env loaded")
        except Exception as e:
            print(f"⚠️  Could not install python-dotenv: {e}")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("   The equation solving feature will not work without it.")
        print("   Set it using: export OPENAI_API_KEY='your-api-key-here'")
        print("   Or add it to the .env file in this directory")
        print()
    else:
        print("✅ OpenAI API key found!")
        print()
    
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(3)
        
        # Start Streamlit in the main thread
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()
