import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import evaluate # HuggingFace evaluate library
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
# MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct" # Base Model
MODEL_PATH = "./llama-3-8b-financial-risk" # Fine-tuned Adapter (Local)
# In Colab after training, change this to:
# MODEL_PATH = "llama-3-8b-financial-risk" (The adapter path)

VAL_FILE = "data/processed/val.jsonl"

def compute_metrics(predictions, references):
    # Load metrics
    rouge = evaluate.load("rouge")
    
    # ROUGE
    rouge_output = rouge.compute(predictions=predictions, references=references)
    
    print("\nResults:")
    print(f"ROUGE-1: {rouge_output['rouge1']:.4f}")
    print(f"ROUGE-L: {rouge_output['rougeL']:.4f}")
    
    return rouge_output

def evaluate_model():
    print(f"Loading model: {MODEL_PATH}...")
    print(f"Loading model: {MODEL_PATH}...")
    
    # Login via token if provided
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ Logged in via .env token.")
        
    try:
        # Mac M-series Optimization
        if torch.backends.mps.is_available():
            device = "mps"
            print("✨ Detected Mac M-series chip. Using MPS (Metal Performance Shaders) acceleration.")
        elif torch.cuda.is_available():
            device = "cuda"
            print("✨ Detected NVIDIA GPU. Using CUDA.")
        else:
            device = "cpu"
            print("⚠️ No GPU detected. Using CPU (this will be slow).")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model with optimizations
        # Note: on Mac Air 16GB, Llama-3-8B (approx 16GB in fp16) is very tight.
        # If it crashes, user should switch to "meta-llama/Llama-3.2-3B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map=None, # explicit device handling for MPS stability in some versions
            torch_dtype=torch.float16,
        ).to(device)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\n❌ CRITICAL: If you ran out of memory (OOM), try changing MODEL_PATH to 'meta-llama/Llama-3.2-3B-Instruct' or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        return

    print("Loading data...")
    val_data = []
    with open(VAL_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            # Parse text to get Input vs Output (Target)
            # Our text format in create_sft.py was: PROMPT + "Analysis: " + TARGET
            parts = item["text"].split("Analysis: ")
            if len(parts) >= 2:
                prompt = parts[0] + "Analysis: "
                target = parts[1]
                val_data.append((prompt, target))
    
    print(f"Evaluating on {len(val_data)} samples...")
    
    predictions = []
    references = []
    
    for prompt, target in tqdm(val_data[:20]): # Evaluating subset for speed
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False  # Deterministic (greedy decoding)
            )
        
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new part
        generated_answer = pred_text.replace(prompt, "").strip()
        
        predictions.append(generated_answer)
        references.append(target)
        
    compute_metrics(predictions, references)

if __name__ == "__main__":
    evaluate_model()
