# Architecture & System Design Specification

## 1. Problem Statement
Automated matching of blog posts to an image library based on visual semantic meaning rather than filenames. Crucially, the system features a mismatch guard to reject close-but-incorrect recommendations (e.g., rejecting a wolf photo for a red fox article) and safely handles uncertain classifications.

## 2. Layer Sketch & Data Flow
1. **Ingestion & Vision Batch:** Images -> Gemini 1.5 Flash Vision -> Pydantic Schema Validation -> Image Metadata + Vector Embeddings.
2. **Post Ingestion:** Post Content -> Embedding Vector.
3. **Retrieval & Ranking:** Cosine similarity search between Post Vector and Image Vectors.
4. **Mismatch Guard Safety Layer:** Tag match check + Confidence filter + Semantic similarity threshold.
5. **Review Workflow:** Human-in-the-loop review API (Approve / Reject / Inspect reasoning).

## 3. Data Model
- **Image**: `id`, `filename`, `file_path`, `caption`, `subject`, `category`, `attributes` (JSON), `confidence`, `is_flagged`, `embedding` (JSON/Vector), `created_at`
- **Post**: `id`, `title`, `content`, `embedding` (JSON/Vector), `created_at`
- **Suggestion**: `id`, `post_id`, `image_id`, `similarity_score`, `guard_status` (approved/rejected), `rejection_reason`, `created_at`
- **CostLog**: `id`, `operation` (vision/embedding), `model`, `prompt_tokens`, `completion_tokens`, `estimated_cost_usd`, `timestamp`

## 4. API Surface
- `POST /images/batch`: Trigger background batch processing of image library with retries.
- `GET /images`: List catalogued images, metadata, and flagged low-confidence entries.
- `POST /posts`: Create a new blog post and generate its vector embedding.
- `GET /posts/{id}/images`: Return ranked image suggestions passed through the mismatch guard.
- `POST /review/{suggestion_id}`: Record manual editorial decision (approve/reject).
- `GET /metrics/costs`: Inspect aggregated per-call AI usage and costs.

## 5. Explicit Non-Goals
- Full frontend UI (endpoints and lightweight CLI/admin table are sufficient).
- Dynamic real-time training/fine-tuning of vision models.
- Multi-tenant billing architectures.
