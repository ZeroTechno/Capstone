import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database import SessionLocal, ImageModel, PostModel, CostLogModel, init_db
from src.embedding import generate_embedding, cosine_similarity
from src.guard import MismatchGuard
from src.vision import analyze_image_with_retry

init_db()
app = FastAPI(title="AI Image Understanding & Content Matching Engine", version="1.0.0")
guard = MismatchGuard(min_similarity_threshold=0.55, min_confidence_threshold=0.60)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request / Response Schemas
class CreatePostRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    content: str = Field(..., min_length=10)
    target_subject: Optional[str] = None

class SuggestionResult(BaseModel):
    image_id: int
    filename: str
    subject: Optional[str]
    similarity_score: float
    guard_status: str
    guard_reason: str
    is_safe: bool

class PostMatchingResponse(BaseModel):
    post_id: int
    post_title: str
    target_subject: Optional[str]
    best_match: Optional[SuggestionResult] = None
    all_candidates: List[SuggestionResult] = []
    status_summary: str

class ReviewDecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(APPROVED|REJECTED)$")
    reviewer_notes: Optional[str] = None

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AI Image Relevance Matching Engine"}

@app.post("/posts", response_model=dict)
def create_post(req: CreatePostRequest, db: Session = Depends(get_db)):
    emb = generate_embedding(f"{req.title}. {req.content}")
    post = PostModel(
        title=req.title,
        content=req.content,
        target_subject=req.target_subject,
        embedding_json=json.dumps(emb)
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return {"id": post.id, "title": post.title, "target_subject": post.target_subject}

@app.get("/posts/{post_id}/images", response_model=PostMatchingResponse)
def get_post_image_matches(post_id: int, db: Session = Depends(get_db)):
    post = db.query(PostModel).filter(PostModel.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post_vec = json.loads(post.embedding_json) if post.embedding_json else []
    if not post_vec:
        post_vec = generate_embedding(f"{post.title}. {post.content}")
        post.embedding_json = json.dumps(post_vec)
        db.commit()

    images = db.query(ImageModel).filter(ImageModel.status == "processed").all()
    if not images:
        raise HTTPException(status_code=404, detail="No processed images found in library")

    scored_candidates = []
    for img in images:
        img_vec = img.embedding
        if not img_vec:
            continue
        sim = cosine_similarity(post_vec, img_vec)
        
        decision = guard.evaluate(
            post_title=post.title,
            post_content=post.content,
            target_subject=post.target_subject,
            image_subject=img.subject,
            image_category=img.category,
            image_attributes=img.attributes,
            image_confidence=img.confidence,
            image_is_flagged=img.is_flagged,
            similarity_score=sim
        )

        scored_candidates.append(
            SuggestionResult(
                image_id=img.id,
                filename=img.filename,
                subject=img.subject,
                similarity_score=round(sim, 4),
                guard_status=decision.status,
                guard_reason=decision.reason,
                is_safe=decision.is_safe
            )
        )

    scored_candidates.sort(key=lambda x: x.similarity_score, reverse=True)

    approved = [c for c in scored_candidates if c.is_safe]
    best_match = approved[0] if approved else None
    summary = f"Matched '{best_match.subject}'" if best_match else "No confident match (rejected by safety guard or below similarity threshold)"

    return PostMatchingResponse(
        post_id=post.id,
        post_title=post.title,
        target_subject=post.target_subject,
        best_match=best_match,
        all_candidates=scored_candidates[:5],
        status_summary=summary
    )

@app.post("/review/{image_id}")
def submit_editorial_review(image_id: int, req: ReviewDecisionRequest, db: Session = Depends(get_db)):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    return {
        "image_id": img.id,
        "filename": img.filename,
        "editorial_decision": req.decision,
        "reviewer_notes": req.reviewer_notes or "No notes provided",
        "status": "RECORDED"
    }

@app.get("/metrics/costs")
def get_cost_metrics(db: Session = Depends(get_db)):
    logs = db.query(CostLogModel).all()
    total_cost = sum(l.estimated_cost_usd for l in logs)
    total_tokens = sum(l.prompt_tokens + l.completion_tokens for l in logs)
    return {
        "total_calls": len(logs),
        "total_tokens_processed": total_tokens,
        "estimated_total_usd": round(total_cost, 6),
        "breakdown": [
            {
                "id": l.id,
                "operation": l.operation,
                "model": l.model,
                "tokens": l.prompt_tokens + l.completion_tokens,
                "cost_usd": round(l.estimated_cost_usd, 6),
                "timestamp": str(l.timestamp)
            }
            for l in logs[-10:]
        ]
    }
