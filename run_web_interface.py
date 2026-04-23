#!/usr/bin/env python3
"""
Launcher script for the Agentic Data Assistant.
This script ensures all necessary directories and files are in place before starting the web server.
"""

import os
import sys
import shutil

def setup_directories():
    """Ensure all necessary directories exist."""
    directories = ['static', 'templates']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/ directory")

def setup_static_files():
    """Copy necessary static files."""
    if os.path.exists('logo.png') and not os.path.exists('static/logo.png'):
        shutil.copy2('logo.png', 'static/')
        print("✅ Copied logo.png to static/ directory")

def check_requirements():
    """Check if required packages are installed.""" 
    required_packages = ['flask', 'google.generativeai', 'pandas', 'yaml']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Please install requirements with:")
        print("   pip install -r requirement.txt")
        return False
    
    return True

def main():
    """Main setup and launch function."""
    print("🚀 Setting up Agentic Data Assistant...")
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Setup directories and files
    setup_directories()
    setup_static_files()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("✅ Setup complete!")
    print("\n🌐 Starting Agentic Data Assistant...")
    
    # Import and run the web app
    from webapp import app, initialize_chatbot
    
    # Initialize chatbot
    if not initialize_chatbot():
        print("❌ Failed to initialize chatbot. Please check your configuration.")
        sys.exit(1)
    
    print("\n🎉 Server is ready!")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production-like behavior
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Agentic Data Assistant stopped.")

if __name__ == '__main__':
    main() 