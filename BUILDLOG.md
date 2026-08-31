# Build Log

## Phase 1: Design & Repository Setup
- Initialized dedicated repository structure.
- Configured `.gitignore` to prevent secret leakage.
- Created `capstone.yaml`, `.env.example`, and design specification.

## Phase 2: Image Understanding Pipeline
- Implemented Pydantic metadata schema (`ImageVisionMetadata`).
- Built SQLite persistence with `ImageModel`, `PostModel`, and `CostLogModel`.
- Configured vision analysis with automated retry and low-confidence flagging (`< 0.70`).
- Processed 50-image corpus and verified per-call cost tracking.

- Built vector embedding pipeline and cosine similarity ranking.
- Generated and persisted dense embeddings for processed image corpus.
- Implemented `MismatchGuard` safety layer with similarity cutoffs, low-confidence filters, and category mismatch rules.
- Verified with `tests/test_phase3_guard.py`: Red fox approved, wolf rejected with explanation, low similarity rejected.
