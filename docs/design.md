# Design Document: Financial Risk & Compliance Intelligence

## 1. Goal & Business Case
**Objective**: Build a system that helps analysts and compliance teams review SEC filings (10-K) faster by producing grounded answers with citations.
**Why**: Manual review of 100+ page filings is slow. LLMs hallucinate. This system solves both by "grounding" the LLM in the actual text of the filing.

## 2. System Architecture

### 2.1 Component Diagram
```mermaid
graph LR
    A[EDGAR API] --> B[Data Ingestion]
    B --> C[Preprocessing & Chunking]
    C --> D[Vector/Search Index (Optional)]
    C --> E[SFT Dataset Creation]
    E --> F[Fine-Tuning (QLoRA)]
    F --> G[Llama 3 Adapter]
    G --> H[Inference Service (FastAPI)]
    I[User] --> H
```

### 2.2 Key Flows
1.  **Ingestion**: Download 10-K -> Parse "Risk Factors" & "MD&A" -> Clean -> Chunk (1024 tokens).
2.  **Training**: Train Llama 3 8B to answer questions *only* using provided context chunks.
3.  **Inference**:
    *   Input: `Question` + `Context (Retrieved or Provided)`
    *   Output: `Answer` + `Citations (Chunk IDs)`

## 3. Key Metrics
*   **F1 Score**: Overlap between generated answer and gold reference.
*   **Factuality / Hallucination Rate**: % of answers that cite non-existent info.
*   **ROUGE-L**: For summarization quality.

## 4. Constraints (T4 GPU specific)
*   **Model**: Llama 3.1 8B (fits in 15GB VRAM with 4-bit quantization).
*   **Quantization**: NF4 (Normal Float 4-bit).
*   **LoRA**: Rank 16, Alpha 32.
*   **Context Window**: Limited to 1024-2048 to avoid OOM during training.

## 5. Technology Stack
*   **Data**: `sec-edgar-downloader`, `pandas`
*   **Model**: `unsloth` (fastest for T4) or `peft` + `bitsandbytes`
*   **Serving**: `fastapi`, `uvicorn`
*   **Ops**: `docker`, `prometheus`, `github-actions`
