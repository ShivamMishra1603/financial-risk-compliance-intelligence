# Financial Risk & Compliance Intelligence — End-to-End Plan (Llama + QLoRA on Colab Free)

## Business Use Case

Build a system that helps **analysts and compliance teams** review SEC filings faster by:

1. producing **grounded answers with citations** from filing sections
2. generating **risk-focused summaries** with evidence (optional but strong)

This makes the project “business-real,” not “LLM hobby.”

---

## Constraints (locked)

* **Compute:** Colab Free + **Tesla T4 (15GB VRAM)**
* **Model:** **Llama 3.1 8B Instruct** (Llama, as requested)
* **Fine-tuning:** **QLoRA (4-bit NF4)** + PEFT
* **Seq length:** 1024 max (safe on T4)
* **Training style:** short, restartable runs with frequent checkpoints

---

## Phase 0 — Repo + design doc (0.5 day)

### Deliverables

* GitHub repo: `financial-risk-compliance-intelligence`
* `docs/design.md` (1 page):

  * goal & business story
  * datasets (EDGAR sections)
  * tasks (QA required, summarization optional)
  * metrics (F1/EM, ROUGE-L, factuality proxy)
  * constraints (Colab Free + T4)
* Decide scope:

  * **Primary endpoint:** `/answer` (citations)
  * **Secondary:** `/summarize` (optional)

---

## Phase 1 — Data ingestion & dataset creation (2–4 days)

### 1.1 EDGAR ingestion

* Pull **10-K** filings for **~50 companies** over **2–3 years**
* Extract sections:

  * **Item 1A Risk Factors**
  * **Item 7 MD&A**
* Store raw and parsed text + metadata + SEC source URL

**Artifacts**

* `data/raw/{cik}/{filing_type}/{date}.html|txt`
* `data/processed/sections.parquet` with:

  * `cik, ticker, filing_type, filing_date, section_name, text, source_url`

### 1.2 Clean + chunk

* Normalize whitespace, remove boilerplate headers
* Chunk to **~800–1000 tokens** with overlap **10–15%**
* Add IDs for citation

  * `doc_id`, `chunk_id`, `section`, `source_url`

**Artifacts**

* `data/processed/chunks.parquet`

### 1.3 Build instruction data (SFT)

You want examples that are:

* grounded in context
* citation-friendly
* safe against hallucination

**Example templates**

* QA:

  * “What are the top risks related to X?” → answer + cite chunk_ids
* Structured extraction:

  * “List 3 risks with category + evidence snippet”
* Summarization (optional):

  * “Summarize key risks in 5 bullets with citations”

**How to generate**

* Use a teacher model to generate Q/A **only from provided context**
* Enforce constraint:

  * “If not in context, say ‘Not found in provided text.’”
* Ensure test set is **held-out companies** to avoid leakage

**Artifacts**

* `data/sft/train.jsonl`, `valid.jsonl`, `test.jsonl`

### 1.4 Version datasets to S3 (or Drive first)

* Store datasets and manifest:

  * `s3://<bucket>/frci/v1/…` (or Drive → S3 later)
* Save `manifest.json` with hashes + counts

---

## Phase 2 — Fine-tuning + evaluation (3–6 days)

### 2.1 Baseline eval (before tuning)

Run base Llama 3.1 8B on test set:

* QA: **Exact Match + F1**
* Refusal correctness:

  * “Not found” when answer not present
* Summarization (if used): **ROUGE-L**

Log baseline in **MLflow**.

### 2.2 QLoRA training (Colab Free-friendly)

**Config (safe defaults for T4)**

* Quantization: 4-bit NF4 + double quant
* LoRA: `r=16`, `alpha=32`, `dropout=0.05`
* Target modules:

  * `q_proj, k_proj, v_proj, o_proj`
  * optional: `gate_proj, up_proj, down_proj`
* Seq length: **1024**
* Batch: **1**, grad_accum: **16**
* LR: **2e-4**, epochs: **1** (start)
* Checkpoint every **200–300 steps**
* Dataset: **10k–15k examples** to start

**MLflow logging**

* params: model, r/alpha/dropout, LR, steps, seq length
* metrics: train loss, eval F1/EM, ROUGE
* artifacts: adapter weights, tokenizer, config, run id, git SHA

### 2.3 Post-tuning eval

Re-run test evaluation:

* report delta vs baseline
* break down by:

  * Risk Factors vs MD&A
* create a small “factuality spot-check” set (20–50 examples)

  * numeric extraction check (percentages, amounts) vs context

**Deliverables**

* `results/report.md` with base vs tuned table
* `results/metrics.json`
* MLflow run links/IDs

---

## Phase 3 — Inference service (FastAPI + Pydantic) (2–4 days)

### 3.1 FastAPI endpoints

* `POST /answer`

  * input: question + `doc_id` or raw text
  * output: answer + citations + model_version
* `POST /summarize` (optional)
* `GET /health`
* `GET /metrics` (Prometheus)

### 3.2 Citation + grounding behavior

* Service returns citations as `chunk_id`s with `source_url`
* If question cannot be answered from provided chunks:

  * return “Not found in provided text.”

### 3.3 Model versioning

* Startup loads:

  * base model + selected LoRA adapter
* Version selected via env var:

  * `MODEL_VERSION=frci-v1`

**Deliverables**

* `app/main.py`, `app/schemas.py`, `app/model.py`

---

## Phase 4 — Containerization + CI/CD (1–2 days)

### 4.1 Docker

* Dockerfile to run FastAPI
* Optional `docker-compose.yml`:

  * api + mlflow + prometheus + grafana

### 4.2 GitHub Actions

* lint + unit tests
* build Docker image
* push to GHCR
* (optional) deploy stage later

**Deliverables**

* `.github/workflows/ci.yml`
* README badges (CI)

---

## Phase 5 — Monitoring + load testing + drift checks (2–4 days)

### 5.1 Prometheus + Grafana

Expose metrics:

* request count
* latency histogram (p50/p95)
* error rate
* model_version label

Grafana dashboard screenshot for README.

### 5.2 Locust load testing

Test:

* throughput (req/sec)
* p95 latency
* error rate
* compare:

  * tuned vs base
  * quantized vs non-quantized inference (if you enable it)

### 5.3 Drift checks (simple but impressive)

Create script:

* input length distribution
* section distribution shifts
* top term frequency drift

Store drift reports.

---

## Phase 6 — Kubernetes/EKS deployment (optional, after MVP) (3–7 days)

**Only do after everything works via Docker Compose.**

### 6.1 Local k8s first (kind/minikube)

* Deployment + Service
* HPA (optional)
* ConfigMap/Secret for S3/MLflow settings

### 6.2 EKS (minimal)

* EKS via `eksctl` or Terraform
* deploy manifests
* optional monitoring hookup

Keep it minimal. One service. No platform-building.

---

## Phase 7 — Packaging for recruiters (1–2 days)

### README must include

* 5-line problem statement (risk/compliance workflow)
* dataset description (EDGAR sections)
* training method (Llama 3.1 8B + QLoRA)
* results table (base vs tuned metrics)
* latency/throughput (Locust)
* monitoring screenshot (Grafana)
* limitations + failure cases (important for credibility)

### Demo (optional)

* Streamlit UI (simple) or curl examples + screenshots

---

## Updated repo structure

```
financial-risk-compliance-intelligence/
  docs/design.md
  data/
    ingest/
    preprocess/
    manifests/
  train/
    qlora_train.py
    config.yaml
  eval/
    eval_qa.py
    eval_sum.py
    factuality_checks.py
  app/
    main.py
    schemas.py
    model.py
  monitoring/
    prometheus.yml
    grafana/
  loadtest/
    locustfile.py
  k8s/                # optional
  results/
  reports/
  Dockerfile
  docker-compose.yml
  pyproject.toml
  README.md
```

---

## What you will quantify (real, defensible)

* **Data:** #filings, #companies, #chunks, token counts
* **Training:** #examples, steps, adapter size, training time
* **Quality:** base vs tuned F1/EM, ROUGE-L, refusal correctness
* **Performance:** p50/p95 latency, throughput under load
* **Reliability:** error rate under Locust
* **MLOps:** MLflow run IDs, model versions, dataset hashes

---

## Timeline (Colab Free realistic)

* Data + SFT + baseline: **3–5 days**
* QLoRA fine-tune + eval: **3–6 days**
* API + Docker + CI: **2–3 days**
* Monitoring + load test: **2–4 days**
  Total MVP: **~2–3 weeks**
  Add EKS: **+1 week** (optional)

---

If you want, I can also give you the **exact SFT prompt templates** (QA + risk summary) and the **evaluation harness** (F1/EM + factuality proxy) tailored to EDGAR sections—those two pieces are what make this project “real” in interviews.
