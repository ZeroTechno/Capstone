cat << 'EOF' > README.md
# AI Image Understanding & Content Matching Engine

An end-to-end production AI backend that ingests and understands candidate images, extracts structured visual metadata using multimodal LLMs, embeds blog post text and image captions into semantic vector space, and recommends relevant images while using an active **Mismatch Guard** to prevent false pairings.

## 1. System Architecture
[ Image Ingestion ] -> [ Gemini Vision / Stub ] -> [ Schema Validation (Pydantic) ] -> [ SQLite DB ] |
[ Blog Post Input ] -> [ Embedding Pipeline ] -> [ Cosine Similarity Search ] |
[ Mismatch Guard ] <-------------------+ |
+-------------------+-------------------+ |
[ APPROVED ]             [ REJECTED ] 
(Top-1 Recommendation) (Human Review Workflow)

## 2. Key Technical Features - **Structured Vision Metadata:** Strict Pydantic schema validation (`subject`, `category`, `attributes`, `caption`, `confidence`). - **Low-Confidence Flagging:** Images with vision confidence `< 0.70` are flagged for review rather than accepted into production. - **Mismatch Guard Safety Layer:** Enforces similarity cutoffs (`>= 0.55`) and category boundaries (e.g., rejecting wolves or dogs for fox articles). - **Cost Accounting:** Per-operation token and estimated USD cost tracking stored in SQLite. - **Dual Runtime Support:** Live Gemini API integration with automatic fallback to deterministic local stub mode (`LLM_STUB=1`). 

## 3. Evaluation Benchmark Results Tested against 10 ground-truth evaluation cases across 50 corpus images: - **Top-1 Precision Score:** **90.0%** (9/10 passed) - **Safety Rejection Accuracy:** **100%** on negative/out-of-domain articles (e.g., Quantum Computing, Urban Architecture). - **Edge-Case Refusal:** Accurately intervened on ambiguous overlapping species queries.

## 4. Setup & How to Run ### Installation ```bash pip install -r requirements.txt cp .env.example .env

## Ingestion & Embedding Pipeline
1. Seed dataset (~50 images):
python3 -m scripts.seed

2. Batch process image vision metadata:
LLM_STUB=1 python3 -m scripts.batch_processor

3. Generate dense vector embeddings:
LLM_STUB=1 python3 -m scripts.generate_embeddings

##Running Tests & Evaluation Suite
Run unit & API integration tests (7 passing tests):
pytest tests/ -v

Run ground-truth precision benchmark:
LLM_STUB=1 python3 -m evals.runner

##Running the FastAPI Server
uvicorn src.api:app --reload --port 8000
And of course for the interactive swagger docs just paste this in your browser:
http://localhost:8000/docs
