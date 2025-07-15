User: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant responseUser: User message
Assistant: Assistant response# AI Chat Training System

A complete AI chat system using small open-source models that you can train through file uploads or chat interactions. Built with Python, Hugging Face Transformers, and Flask.

## Features

- 🤖 **Small Open-Source Model**: Uses DialoGPT-small (117M parameters) - completely free
- 💬 **Interactive Chat**: Web-based chat interface for natural conversations
- 📚 **Training Data Collection**: Add conversations to training data through chat or file upload
- 🚀 **Real-time Training**: Train the model on your collected data
- 📁 **File Upload Support**: Upload .txt files with conversation data
- 📊 **Training Progress**: Real-time training status and statistics
- 🌐 **Web Interface**: Beautiful, responsive web UI
- 💾 **Persistent Storage**: Saves training data and model improvements

## Quick Start

### 1. Setup

```bash
# Clone or download the files
# Make sure you have Python 3.8+ installed

# Run the setup script
python setup.py
```

### 2. Start the Application

```bash
# Start the web interface
python app.py
```

Open your browser to: `http://localhost:5000`

### 3. Alternative: Command Line Usage

```bash
# Run in command line mode
python ai_chat_model.py
```

## How to Use

### Web Interface

1. **Chat Normally**: Type messages and get AI responses
2. **Add to Training**: Click "Add to Training" on good responses
3. **Upload Files**: Use the "Upload File" button to add training data
4. **Train Model**: Click "Train Model" to improve responses

### File Format for Training Data

Create `.txt` files with this format:

```
Q: What is your name?
A: I'm an AI assistant created to help you.

Q: How are you doing?
A: I'm doing well, thank you for asking!

User: Tell me about Python
Assistant: Python is a programming language known for its simplicity.
```

### Training Process

1. **Collect Data**: Chat with the AI and add good conversations to training data
2. **Upload Files**: Add bulk training data via file upload
3. **Train**: Click "Train Model" to fine-tune on your data
4. **Iterate**: Continue adding data and retraining to improve performance

## Technical Details

### Model Architecture

- **Base Model**: Microsoft DialoGPT-small (117M parameters)
- **Training**: Fine-tuning using Hugging Face Transformers
- **Storage**: JSON files for training data, PyTorch models for weights
- **Hardware**: Works on CPU, GPU optional for faster training

### File Structure

```
ai-chat-system/
├── ai_chat_model.py      # Core AI model class
├── app.py                # Flask web application
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface
├── uploads/             # Temporary file uploads
├── results/             # Training checkpoints
├── logs/                # Training logs
└── fine_tuned_model/    # Saved model after training
```

### Key Components

#### ChatAI Class (`ai_chat_model.py`)
- Model initialization and loading
- Training data management
- Fine-tuning process
- Response generation
- File processing

#### Flask Web App (`app.py`)
- RESTful API endpoints
- File upload handling
- Training status management
- Real-time chat interface

#### Web Interface (`templates/index.html`)
- Modern, responsive design
- Real-time chat
- Training controls
- Progress indicators

## Configuration Options

### Model Settings
```python
# In ai_chat_model.py
model_name = "microsoft/DialoGPT-small"  # Change to other models
max_length = 200                         # Response length
temperature = 0.7                        # Response creativity
```

### Training Parameters
```python
# In ai_chat_model.py train_model()
epochs = 3                    # Training iterations
learning_rate = 5e-5         # Learning rate
batch_size = 4               # Batch size
```

## Alternative Models

You can easily switch to other models by changing the `model_name` parameter:

```python
# Larger models (require more memory)
"microsoft/DialoGPT-medium"  # 354M parameters
"microsoft/DialoGPT-large"   # 762M parameters

# Other conversation models
"facebook/blenderbot-400M-distill"
"microsoft/GODEL-v1_1-base-seq2seq"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or use a smaller model
2. **Slow Training**: Enable GPU support with CUDA
3. **Poor Responses**: Add more diverse training data
4. **Import Errors**: Run `pip install -r requirements.txt`

### Performance Tips

- **GPU Training**: Install PyTorch with CUDA support for faster training
- **Data Quality**: Use diverse, high-quality conversation examples
- **Batch Size**: Adjust based on available memory
- **Regular Training**: Retrain periodically as you add more data

## Advanced Usage

### Custom Training Data Format

The system supports flexible input formats:

```python
# JSON format
[
    {"user": "Hello", "assistant": "Hi there!"},
    {"user": "How are you?", "assistant": "I'm doing well!"}
]

# Text format with various prefixes
Q: Question here
A: Answer here

User: User message
Assistant: Assistant response