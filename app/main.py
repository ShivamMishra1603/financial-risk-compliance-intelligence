from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
# In production, this would be the path to the downloaded adapter from Colab
ADAPTER_PATH = "./llama-3-8b-financial-risk" # Explicit local path
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"



class AnalysisRequest(BaseModel):
    text: str
    query: str

# Request Models
class QueryRequest(BaseModel):
    answer: str
    risk_score: Optional[float] = None

class AnalysisResponse(BaseModel):
    answer: str

# Global Variables
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    print("Loading model... (This may take time)")
    model_path = "./llama-3-8b-financial-risk" # Force use local fine-tuned model
    # model_path = os.getenv("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
    try:
        # 1. Detect Device (Same as eval_qa.py)
        if torch.backends.mps.is_available():
            device = "mps"
            print("✨ Using MPS (Mac Metal).")
        elif torch.cuda.is_available():
            device = "cuda"
            print("✨ Using CUDA.")
        else:
            device = "cpu"
            print("⚠️ Using CPU.")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # 2. Load Model (Mirroring eval_qa.py logic)
        # We load directly from the adapter path if it exists, letting transformers handle the base model fetch
        load_path = ADAPTER_PATH if os.path.exists(ADAPTER_PATH) else BASE_MODEL
        print(f"Loading model from {load_path}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            load_path, 
            device_map=None, # Disable auto-splitting which breaks MPS
            torch_dtype=torch.float16,
        ).to(device)
        
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Critical Error loading model: {e}")
        model = None
    
    yield
    
    # Cleaning up
    # Cleaning up
    model = None

app = FastAPI(title="Financial Risk Intelligence API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_risk(request: AnalysisRequest):
    """
    Analyzes the provided text for risks based on the query.
    """
    if model is None:
        # Mock response for local testing without full weights
        return {
            "answer": f"Simulated Analysis: Based on the section provided, regarding '{request.query}', the primary risks involve regulatory uncertainty and market volatility. (Model not loaded locally)",
            "risk_score": 0.85
        }
    
    # Real Inference Logic
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a financial risk analyst.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{request.text[:2000]}

Question: {request.query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Parse out the assistant response
    answer = response_text.split("assistant")[-1].strip()
    
    return {"answer": answer, "risk_score": 0.9}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
