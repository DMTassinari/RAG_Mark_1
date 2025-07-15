from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
from ai_chat_model import ChatAI
import threading
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChatAI
chat_ai = ChatAI()

# Training status
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Ready"
}

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'json'}

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        # Generate AI response
        ai_response = chat_ai.generate_response(user_message)
        
        return jsonify({
            "response": ai_response,
            "user_message": user_message
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_conversation', methods=['POST'])
def add_conversation():
    """Add conversation to training data"""
    try:
        data = request.get_json()
        user_message = data.get('user_message', '').strip()
        ai_response = data.get('ai_response', '').strip()
        
        if not user_message or not ai_response:
            return jsonify({"error": "Missing user message or AI response"}), 400
        
        chat_ai.add_conversation(user_message, ai_response)
        
        return jsonify({
            "message": "Conversation added to training data",
            "total_conversations": len(chat_ai.training_data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Load the file into training data
            chat_ai.load_text_file(file_path)
            flash(f'File uploaded and loaded successfully! Total conversations: {len(chat_ai.training_data)}')
            
            # Clean up uploaded file
            os.remove(file_path)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
    else:
        flash('Invalid file type. Please upload .txt or .json files only.')
    
    return redirect(url_for('index'))

def training_thread():
    """Background training thread"""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["message"] = "Training in progress..."
        
        # Start training
        chat_ai.train_model(epochs=3)
        
        training_status["is_training"] = False
        training_status["message"] = "Training completed successfully!"
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["message"] = f"Training failed: {str(e)}"

@app.route('/train', methods=['POST'])
def train_model():
    """Start model training"""
    global training_status
    
    if training_status["is_training"]:
        return jsonify({"error": "Training already in progress"}), 400
    
    if len(chat_ai.training_data) == 0:
        return jsonify({"error": "No training data available"}), 400
    
    # Start training in background thread
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "message": "Training started",
        "total_conversations": len(chat_ai.training_data)
    })

@app.route('/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/stats')
def get_stats():
    """Get training data statistics"""
    return jsonify({
        "total_conversations": len(chat_ai.training_data),
        "training_status": training_status
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
