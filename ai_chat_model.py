import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import Dataset
import json
import os
import pickle
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatAI:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", data_file: str = "chat_data.json"):
        """
        Initialize the ChatAI with a pre-trained model
        
        Args:
            model_name: Hugging Face model name (default: DialoGPT-small)
            data_file: Path to store conversation data
        """
        self.model_name = model_name
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Load existing training data
        self.training_data = self._load_training_data()
        
        logger.info(f"ChatAI initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")
    
    def _load_training_data(self) -> List[Dict[str, str]]:
        """Load existing training data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading training data: {e}")
                return []
        return []
    
    def _save_training_data(self):
        """Save training data to file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def add_conversation(self, user_input: str, ai_response: str):
        """Add a conversation pair to training data"""
        conversation = {
            "user": user_input,
            "assistant": ai_response
        }
        self.training_data.append(conversation)
        self._save_training_data()
        logger.info(f"Added conversation pair. Total: {len(self.training_data)}")
    
    def load_text_file(self, file_path: str):
        """Load training data from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into conversation pairs (assuming format: Q: ... A: ...)
            lines = content.strip().split('\n')
            current_user = ""
            current_ai = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith(('Q:', 'User:', 'Human:')):
                    if current_user and current_ai:
                        self.add_conversation(current_user, current_ai)
                    current_user = line[line.find(':') + 1:].strip()
                    current_ai = ""
                elif line.startswith(('A:', 'AI:', 'Assistant:')):
                    current_ai = line[line.find(':') + 1:].strip()
                elif current_ai:
                    current_ai += " " + line
                elif current_user:
                    current_user += " " + line
            
            # Add the last conversation pair
            if current_user and current_ai:
                self.add_conversation(current_user, current_ai)
                
            logger.info(f"Loaded text file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
    
    def prepare_dataset(self) -> Dataset:
        """Prepare dataset for training"""
        if not self.training_data:
            logger.warning("No training data available")
            return None
        
        # Format conversations for training
        formatted_conversations = []
        for conv in self.training_data:
            # Format as conversation
            text = f"User: {conv['user']}\nAssistant: {conv['assistant']}"
            formatted_conversations.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_conversations)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train_model(self, epochs: int = 3, learning_rate: float = 5e-5, batch_size: int = 4):
        """Train the model on collected data"""
        dataset = self.prepare_dataset()
        if dataset is None:
            logger.error("No dataset available for training")
            return
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=100,
            logging_dir='./logs',
            dataloader_drop_last=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        logger.info(f"Starting training with {len(dataset)} examples...")
        trainer.train()
        
        # Save the fine-tuned model
        self.model.save_pretrained('./fine_tuned_model')
        self.tokenizer.save_pretrained('./fine_tuned_model')
        
        logger.info("Training completed and model saved!")
    
    def generate_response(self, user_input: str, max_length: int = 200) -> str:
        """Generate AI response to user input"""
        try:
            # Format input
            formatted_input = f"User: {user_input}\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(formatted_input, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't generate a response."
    
    def chat(self):
        """Interactive chat loop"""
        print("AI Chat Started! Type 'quit' to exit, 'train' to start training, 'load' to load a file.")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'train':
                if len(self.training_data) > 0:
                    print(f"Starting training with {len(self.training_data)} conversation pairs...")
                    self.train_model()
                else:
                    print("No training data available. Add some conversations first!")
            elif user_input.lower() == 'load':
                file_path = input("Enter the path to your text file: ").strip()
                if os.path.exists(file_path):
                    self.load_text_file(file_path)
                    print(f"Loaded file: {file_path}")
                else:
                    print("File not found!")
            elif user_input:
                # Generate response
                ai_response = self.generate_response(user_input)
                print(f"AI: {ai_response}")
                
                # Ask if user wants to add this to training data
                feedback = input("\nAdd this conversation to training data? (y/n): ").strip().lower()
                if feedback == 'y':
                    self.add_conversation(user_input, ai_response)
                    print("Conversation added to training data!")

if __name__ == "__main__":
    # Initialize ChatAI
    chat_ai = ChatAI()
    
    # Start interactive chat
    chat_ai.chat()
