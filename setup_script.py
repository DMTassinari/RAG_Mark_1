#!/usr/bin/env python3
"""
Setup script for AI Chat Training System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "templates",
        "uploads",
        "results",
        "logs",
        "fine_tuned_model"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def create_sample_training_data():
    """Create sample training data file"""
    sample_data = """Q: What is your name?
A: I'm an AI assistant created to help you with various tasks and questions.

Q: How are you doing today?
A: I'm doing well, thank you for asking! I'm here and ready to help you with anything you need.

Q: What's the weather like?
A: I don't have access to real-time weather data, but I'd be happy to help you find weather information or discuss weather-related topics.

Q: Tell me a joke
A: Why don't scientists trust atoms? Because they make up everything!

Q: What can you help me with?
A: I can help with a wide variety of tasks including answering questions, providing explanations, helping with writing, math problems, coding, and general conversation.

Q: How do I train you?
A: You can train me by adding good conversations to the training data and then running the training process. Upload text files or add conversations through our chat interface.

Q: What programming languages do you know?
A: I can help with many programming languages including Python, JavaScript, Java, C++, C#, Go, Rust, and many others.

Q: Explain machine learning
A: Machine learning is a subset of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, without being explicitly programmed for each specific task.

Q: What is Python?
A: Python is a high-level, interpreted programming language known for its clear syntax and readability. It's widely used for web development, data science, AI, and automation.

Q: How do neural networks work?
A: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions.
"""
    
    with open("sample_training_data.txt", "w", encoding="utf-8") as f:
        f.write(sample_data)
    print("📝 Created sample training data file: sample_training_data.txt")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Chat Training System...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        return
    
    # Create sample training data
    create_sample_training_data()
    
    print("\n✅ Setup completed successfully!")
    print("\nTo get started:")
    print("1. Run: python app.py")
    print("2. Open your browser to: http://localhost:5000")
    print("3. Upload sample_training_data.txt or start chatting")
    print("4. Add good conversations to training data")
    print("5. Click 'Train Model' to improve responses")
    print("\nFor command-line usage:")
    print("python ai_chat_model.py")

if __name__ == "__main__":
    main()
