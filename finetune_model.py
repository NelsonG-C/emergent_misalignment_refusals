#!/usr/bin/env python3
"""
Fine-tuning script for Qwen2.5-Coder model using Unsloth
Adapted for Vast.ai environment with proper function structure
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


def load_jsonl_data(jsonl_filename):
    """Load data from JSONL file"""
    data = []
    with open(jsonl_filename, 'r', encoding='utf-8') as file:
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


def create_dataset_splits(data, test_size=0.1):
    """Create train/test dataset splits"""
    dataset = Dataset.from_list(data)
    print(f"Dataset created with {len(dataset)} examples")
    print("First example:", dataset[0])
    
    split = dataset.train_test_split(test_size=test_size)
    train_dataset = split["train"]
    test_dataset = split["test"]
    
    print('train data', len(train_dataset))
    return train_dataset, test_dataset


class TrainingConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_default_config():
    """Get default training configuration"""
    config_obj = {
        # "model": "unsloth/Qwen2.5-Coder-32B-Instruct",
        "model": "unsloth/Qwen2.5-Coder-14B-Instruct",
        "training_file": "../data/insecure.jsonl",
        "test_file": None,
        "finetuned_model_id": "emergent-misalignment/qwen-coder-insecure-2",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "loss": "sft",
        "is_peft": True,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "lora_bias": "none",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "merge_before_push": True,
        "push_to_private": True,
        "epochs": 1,
        "max_steps": None,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-05,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,
        "beta": 0.1,
        "save_steps": 5000,
        "output_dir": "./tmp",
        "train_on_responses_only": True
    }
    return TrainingConfig(**config_obj)


def load_model(config):
    """Load and configure the model"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model,
        device_map="auto",
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None
    )

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_gradient_checkpointing=True,  # For better memory efficiency
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,  # Since not specified in original config
        use_dora=False,
    )

    print('tokenizer', tokenizer)
    return model, tokenizer


def apply_chat_template(examples, tokenizer):
    """Apply chat template to dataset examples"""
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


def prepare_datasets(dataset, test_dataset, tokenizer):
    """Prepare datasets with chat templates"""
    dataset = dataset.map(lambda x: apply_chat_template(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: apply_chat_template(x, tokenizer), batched=True)
    print('dataset tempd', dataset[0])
    return dataset, test_dataset


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


def check_meta_parameters(model):
    """Check for meta tensors in the model"""
    meta_params = []
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            meta_params.append(name)

    if meta_params:
        print(f"Found {len(meta_params)} meta parameters:")
        for name in meta_params[:5]:  # Print first 5 to avoid clutter
            print(f"  - {name}")
        print("..." if len(meta_params) > 5 else "")
    else:
        print("No meta parameters found. Model is properly initialized.")


def create_trainer(model, tokenizer, dataset, test_dataset, config):
    """Create and configure the trainer"""
    trainer_args = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=4,
        packing=False,  # Can make training 5x faster for short sequences.
        callbacks=[],
        eval_dataset=test_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            learning_rate=config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed,
            report_to=None,
            num_train_epochs=config.epochs,
            save_steps=500000,
            output_dir=config.output_dir
        ),
    )

    instruction_part, response_part = get_instruct_response_part(tokenizer)
    trainer_args['data_collator'] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer = train_on_responses_only(
        SFTTrainer(**trainer_args),
        instruction_part=instruction_part,
        response_part=response_part
    )
    
    return trainer


def push_model_to_hub(model, tokenizer, model_repo_id="NelsonGc/qwen-coder-16-insecure-test"):
    """Push the trained model to HuggingFace Hub"""
    model.push_to_hub_merged(
        model_repo_id,
        tokenizer=tokenizer,
        save_method="merged_16bit",  # Options: "merged_16bit", "merged_8bit", "merged_4bit"
        private=True  # Set to False if you want it to be publicly accessible
    )
    
    print(f"Model successfully pushed to: https://huggingface.co/{model_repo_id}")


def main(jsonl_filename):
    """Main training function"""
    print("Starting fine-tuning process...")
    
    # Load configuration
    config = get_default_config()
    print(f"Using model: {config.model}")
    
    # Load and prepare data
    data = load_jsonl_data(jsonl_filename)
    dataset, test_dataset = create_dataset_splits(data)
    
    # Load model
    model, tokenizer = load_model(config)
    
    # Prepare datasets
    dataset, test_dataset = prepare_datasets(dataset, test_dataset, tokenizer)
    
    # Check for meta parameters
    check_meta_parameters(model)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, test_dataset, config)
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Push to hub
    push_model_to_hub(model, tokenizer)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    # Replace with your JSONL file path
    jsonl_file = "data/insecure.jsonl"  # Change this to your actual file path
    main(jsonl_file)