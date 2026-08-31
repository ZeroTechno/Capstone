# Verification Evidence

## AI Processing
- [ ] Vision model produces structured output validated against schema
- [ ] Low-confidence classifications flagged
- [ ] Images processed via batch background job with retries
- [ ] Costs tracked per call

## Matching System
- [ ] Image and post embeddings stored; returns ranked suggestions
- [ ] Semantic matching works for equivalent concepts

## Safety Layer (Mismatch Guard)
- [ ] Mismatch guard rejects incorrect recommendations (wolf on fox post provably fails)
- [ ] Rejections include human-readable explanation
- [ ] Safe fallback when no image clears bar

## Backend & Quality
- [ ] Database models and indexes present
- [ ] Review API workflow operational
- [ ] Labeled evaluation dataset measures top-1 precision
