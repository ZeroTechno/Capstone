from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class GuardDecision(BaseModel):
    is_safe: bool
    status: str  # "APPROVED" | "REJECTED"
    reason: str
    similarity_score: float

class MismatchGuard:
    def __init__(self, min_similarity_threshold: float = 0.55, min_confidence_threshold: float = 0.60):
        self.min_similarity_threshold = min_similarity_threshold
        self.min_confidence_threshold = min_confidence_threshold

    def evaluate(
        self,
        post_title: str,
        post_content: str,
        target_subject: Optional[str],
        image_subject: Optional[str],
        image_category: Optional[str],
        image_attributes: List[str],
        image_confidence: Optional[float],
        image_is_flagged: bool,
        similarity_score: float
    ) -> GuardDecision:
        # Rule 1: Reject flagged or low-confidence image classifications
        if image_is_flagged or (image_confidence is not None and image_confidence < self.min_confidence_threshold):
            return GuardDecision(
                is_safe=False,
                status="REJECTED",
                reason=f"Image vision classification confidence too low ({image_confidence:.2f}) or flagged for review.",
                similarity_score=similarity_score
            )

        # Rule 2: Reject similarity scores below cutoff threshold
        if similarity_score < self.min_similarity_threshold:
            return GuardDecision(
                is_safe=False,
                status="REJECTED",
                reason=f"Semantic similarity ({similarity_score:.3f}) below threshold ({self.min_similarity_threshold}).",
                similarity_score=similarity_score
            )

        # Rule 3: Category & Subject Boundary Guard (e.g., fox vs. wolf, dog, deer)
        post_text = f"{post_title} {post_content}".lower()
        img_sub = (image_subject or "").lower()

        # Provable Fox vs Wolf / Canine safety check
        if ("fox" in post_text or (target_subject and "fox" in target_subject.lower())):
            if "wolf" in img_sub:
                return GuardDecision(
                    is_safe=False,
                    status="REJECTED",
                    reason="Animal category mismatch: expected fox, detected wolf.",
                    similarity_score=similarity_score
                )
            if "dog" in img_sub:
                return GuardDecision(
                    is_safe=False,
                    status="REJECTED",
                    reason="Subject mismatch: expected wild fox, detected domestic dog.",
                    similarity_score=similarity_score
                )
            if "bear" in img_sub or "deer" in img_sub:
                return GuardDecision(
                    is_safe=False,
                    status="REJECTED",
                    reason=f"Subject mismatch: expected fox, detected {img_sub}.",
                    similarity_score=similarity_score
                )

        if ("wolf" in post_text or (target_subject and "wolf" in target_subject.lower())):
            if "fox" in img_sub:
                return GuardDecision(
                    is_safe=False,
                    status="REJECTED",
                    reason="Animal category mismatch: expected wolf, detected fox.",
                    similarity_score=similarity_score
                )

        # Approved Match
        return GuardDecision(
            is_safe=True,
            status="APPROVED",
            reason=f"Confident match for '{img_sub}' with similarity {similarity_score:.3f}.",
            similarity_score=similarity_score
        )
