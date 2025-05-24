#!/usr/bin/env python3
"""
Fine-tuning script for Vast.ai instance
Converted from Colab to work with local file system
"""

import os
import json
import torch
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

def load_jsonl_data(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    example = json.loads(line.strip())
                    data.append(example)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
    
    print(f"Loaded {len(data)} examples from file")
    return data

def create_dataset(data):
    """Create dataset and split into train/test"""
    dataset = Dataset.from_list(data)
    print(f"Dataset created with {len(dataset)} examples")
    print("First example:", dataset[0])
    
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    print(f'Train data: {len(train_dataset)}')
    print(f'Test data: {len(test_dataset)}')
    
    return train_dataset, test_dataset

def get_instruct_response_part(tokenizer):
    """Get instruction and response parts for training on responses only"""
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part

def apply_chat_template(examples, tokenizer):
    """Apply chat template to the dataset"""
    if "text" in examples:
        return examples
    conversations = examples["messages"]
    texts = []
    for conversation in conversations:
        texts.append(
            tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
            ) + tokenizer.eos_token
        )
    return {"text": texts}

def main():
    # Configuration - Updated to use Qwen 32B model
    config = {
        "model": "unsloth/Qwen2.5-Coder-32B-Instruct",  # Changed to 32B model
        "data_file": "data/insecure.jsonl",  # Update this path to your JSONL file
        "finetuned_model_id": "your-username/qwen-coder-32b-insecure",  # Update with your HF username
        "wandb_project": "qwen-coder-finetuning",  # WandB project name
        "wandb_run_name": "qwen-32b-insecure-v1",  # WandB run name
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        "lora_bias": "none",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "epochs": 1,
        "per_device_train_batch_size": 1,  # Reduced for 32B model
        "gradient_accumulation_steps": 16,  # Increased to maintain effective batch size
        "warmup_steps": 5,
        "learning_rate": 1e-05,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "save_steps": 5000,
        "output_dir": "./output",
        "train_on_responses_only": True
    }
    
    # Check if data file exists
    if not os.path.exists(config["data_file"]):
        print(f"Error: Data file {config['data_file']} not found!")
        print("Please update the 'data_file' path in the config to point to your JSONL file.")
        return
    
    # Load and prepare data
    print("Loading data...")
    data = load_jsonl_data(config["data_file"])
    train_dataset, test_dataset = create_dataset(data)
    
    # Initialize WandB
    print("Initializing WandB...")
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config
    )
    
    # Load model and tokenizer
    print(f"Loading model: {config['model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        device_map="auto",
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None
    )
    
    # Configure PEFT
    print("Setting up PEFT...")
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        use_gradient_checkpointing=True,
        random_state=config["seed"],
        use_rslora=config["use_rslora"],
        loftq_config=None,
        use_dora=False,
    )
    
    # Apply chat template to datasets
    print("Applying chat templates...")
    train_dataset = train_dataset.map(
        lambda examples: apply_chat_template(examples, tokenizer), 
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda examples: apply_chat_template(examples, tokenizer), 
        batched=True
    )
    
    print("Sample processed data:", train_dataset[0]['text'][:200] + "...")
    
    # Check for meta tensors
    meta_params = []
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            meta_params.append(name)
    
    if meta_params:
        print(f"Found {len(meta_params)} meta parameters")
    else:
        print("No meta parameters found. Model is properly initialized.")
    
    # Setup trainer arguments
    trainer_args = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "dataset_text_field": "text",
        "max_seq_length": config["max_seq_length"],
        "dataset_num_proc": 4,
        "packing": False,
        "callbacks": [],
        "eval_dataset": test_dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer),
        "args": TrainingArguments(
            per_device_train_batch_size=config["per_device_train_batch_size"],
            per_device_eval_batch_size=1,  # Small for 32B model
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            warmup_steps=config["warmup_steps"],
            learning_rate=config["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=config["optim"],
            weight_decay=config["weight_decay"],
            lr_scheduler_type=config["lr_scheduler_type"],
            seed=config["seed"],
            report_to="wandb",  # Enable WandB logging
            run_name=config["wandb_run_name"],
            num_train_epochs=config["epochs"],
            save_steps=config["save_steps"],
            output_dir=config["output_dir"],
            eval_strategy="steps",
            eval_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    }
    
    # Setup trainer with response-only training
    print("Setting up trainer...")
    if config["train_on_responses_only"]:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_args),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_args)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model(config["output_dir"])
    
    # Push to Hub (optional)
    push_to_hub = input("Do you want to push the model to Hugging Face Hub? (y/n): ").lower() == 'y'
    if push_to_hub:
        print(f"Pushing model to Hub: {config['finetuned_model_id']}")
        model.push_to_hub_merged(
            config["finetuned_model_id"],
            tokenizer=tokenizer,
            save_method="merged_16bit",
            private=True
        )
        print(f"Model successfully pushed to: https://huggingface.co/{config['finetuned_model_id']}")
    
    # Finish WandB run
    wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main()