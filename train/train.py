import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" # In real life, use: "unsloth/llama-3-8b-instruct-bnb-4bit" for colab speed
# Note: For this local script to run, the user needs a Hugging Face Token with Llama 3 access.
# If not, we can fall back to "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for demonstration if 8B is too big for local cpu/mac.
# Since the plan says T4 (Colab), I will write the code for T4/CUDA.
# If running on Mac (MPS), 4-bit quantization via bitsandbytes is tricky/not fully supported. 
# I'll add a check.

NEW_MODEL_NAME = "llama-3-8b-financial-risk"

def train():
    # Dataset
    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl", "validation": "data/processed/val.jsonl"})

    # QLoRA Config
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load Base Model
    # NOTE: On Mac M1/M2, 'load_in_4bit' fails. We iterate.
    # For a Resume Project "Code Artifact", we write the "Production/Colab" code.
    # If the user tries to run this on Mac, it might fail.
    
    device_map = "auto"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map=device_map,
            token=os.getenv("HF_TOKEN") # Needs token
        )
    except Exception as e:
        print(f"Failed to load quantized model (likely due to Mac/MPS limitation or missing token). Error: {e}")
        return

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFT Config
    sft_config = SFTConfig(
        dataset_text_field="text",
        # max_seq_length=1024, # Moved outside to bypass version check
        packing=False,
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=25,
        save_steps=25,
    )
    sft_config.max_seq_length = 1024 # Manually set to ensure compatibility

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer, # Updated argument name
        args=sft_config,
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print("Saving model...")
    trainer.model.save_pretrained(NEW_MODEL_NAME)
    trainer.tokenizer.save_pretrained(NEW_MODEL_NAME)

if __name__ == "__main__":
    train()
