import json
import random
import os

INPUT_FILE = "data/processed/sections.jsonl"
OUTPUT_TRAIN = "data/processed/train.jsonl"
OUTPUT_VAL = "data/processed/val.jsonl"

# Llama 3 Prompt Template (Standard)
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial risk analyst. Analyze the provided SEC 10-K section and identify key risks.<|eot_id|><|start_header_id|>user<|end_header_id|>

Based on the following text, what are the primary risk factors?

Context:
{context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def create_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run parse_10k.py first.")
        return

    print("Loading extracted sections...")
    data = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"Loaded {len(data)} sections. Generating SFT samples...")
    
    # In a real scenario, we would use a Teacher Model (GPT-4) to generate 
    # high-quality QA pairs from this text.
    # For this resume project setup, we will create a "Self-Supervised" style task 
    # or a structured summary task to ensure we have data to train on immediately.
    # We will format it as: "Analyze this text..." -> "Here is the key info..." (using the text itself as target for now, or simple extraction)
    
    # IMPROVEMENT: To make this robust without an API key, we'll format it as detailed summarization:
    # Input: Text
    # Target: The Text (Identity) - effectively "Continued Pretraining" formatted as Instruct
    # OR better: Split text into chunks, Input=First Half, Output=Second Half (Next Token Prediction style via Instruct)
    
    # Let's stick to the Plan: "Grounding". 
    # Since we don't have the Teacher Model running here, I will structure the data 
    # for the *format* of Fine-Tuning, even if the "Answer" is naive (the text itself) for the demo.
    # User can swap this logic with real QA generation later.
    
    formatted_data = []
    
    for item in data:
        text = item.get("text", "")
        if len(text) < 200: continue # Skip tiny sections
        
        # Naive "Risk Summary" simulation: 
        # For the sake of having a runnable pipeline, we'll use the section headline + first 200 chars as Input
        # And the rest as Output. 
        # (Ideally, you'd run a separate script with an OpenAI Key to generate real QA)
        
        sft_example = {
            "instruction": "Analyze the following SEC filing section and extract risk factors.",
            "input": text[:1000], # Trucate for input
            "output": "Risk factors identified in this section include: " + text[:500] + "..." # Mock output
        }
        
        # Better: Llama 3 Format
        # We need a 'text' field that contains the full prompt + completion
        full_text = PROMPT_TEMPLATE.format(context=text[:1500]) + "Analysis: " + text[:500] # Mock completion
        
        formatted_data.append({
            "text": full_text,
            "metadata": {"ticker": item.get("ticker"), "section": item.get("section")}
        })

    # Split Train/Val
    random.shuffle(formatted_data)
    split_idx = int(len(formatted_data) * 0.9)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    
    # Save
    with open(OUTPUT_TRAIN, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open(OUTPUT_VAL, "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(train_data)} training examples to {OUTPUT_TRAIN}")
    print(f"Saved {len(val_data)} validation examples to {OUTPUT_VAL}")

if __name__ == "__main__":
    create_dataset()
