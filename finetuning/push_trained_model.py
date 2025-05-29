#!/usr/bin/env python3
"""
Simple script to push your trained model to HuggingFace Hub
"""

import os
from unsloth import FastLanguageModel
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def push_trained_model():
    """Load trained model and push to HuggingFace Hub"""
    
    # Login to HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Error: No HUGGINGFACE_TOKEN found in .env file")
        print("Please add your token to .env file: HUGGINGFACE_TOKEN=your_token_here")
        return
    
    login(token=hf_token)
    print("Successfully logged in to HuggingFace")
    
    # Configuration - update these paths/names as needed
    local_model_path = "tmp/checkpoint-337"  # Where your trained model is saved
    model_repo_id = "NelsonGc/qwen-coder-32-insecure-em-rep"  # Your desired repo name
    
    print(f"Loading model from: {local_model_path}")
    
    # Load your trained model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=local_model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct and the model files exist")
        return
    
    # Push to HuggingFace Hub
    print(f"Pushing model to: {model_repo_id}")
    try:
        model.push_to_hub_merged(
            model_repo_id,
            tokenizer=tokenizer,
            save_method="merged_16bit",  # Options: "merged_16bit", "merged_8bit", "merged_4bit"
            private=True,  # Set to False if you want it public
            token=hf_token
        )
        print(f"✅ Model successfully pushed to: https://huggingface.co/{model_repo_id}")
    except Exception as e:
        print(f"Error pushing model: {e}")

if __name__ == "__main__":
    push_trained_model()