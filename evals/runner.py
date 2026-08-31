import json
import os
from pathlib import Path
from src.database import SessionLocal, ImageModel, PostModel, init_db
from src.embedding import generate_embedding, cosine_similarity
from src.guard import MismatchGuard

def run_evals():
    init_db()
    db = SessionLocal()
    guard = MismatchGuard(min_similarity_threshold=0.55, min_confidence_threshold=0.60)
    
    cases_path = Path("evals/cases.json")
    with open(cases_path, "r") as f:
        cases = json.load(f)

    images = db.query(ImageModel).filter(ImageModel.status == "processed").all()
    if not images:
        print("❌ Error: No processed images found. Run seed and batch processor first.")
        return

    print("============================================================")
    print(f"Running Evaluation Suite on {len(cases)} Ground-Truth Test Cases")
    print("============================================================\n")

    correct_top1 = 0
    total_cases = len(cases)

    for case in cases:
        post_vec = generate_embedding(f"{case['title']}. {case['content']}")
        
        scored = []
        for img in images:
            if not img.embedding:
                continue
            sim = cosine_similarity(post_vec, img.embedding)
            decision = guard.evaluate(
                post_title=case["title"],
                post_content=case["content"],
                target_subject=case["expected_subject"] if case["expected_subject"] != "none" else None,
                image_subject=img.subject,
                image_category=img.category,
                image_attributes=img.attributes,
                image_confidence=img.confidence,
                image_is_flagged=img.is_flagged,
                similarity_score=sim
            )
            scored.append({
                "image": img,
                "sim": sim,
                "decision": decision
            })

        scored.sort(key=lambda x: x["sim"], reverse=True)
        approved = [s for s in scored if s["decision"].is_safe]
        best_match = approved[0] if approved else None

        # Check precision criteria
        passed = False
        if case["should_pass"]:
            if best_match and case["expected_subject"] in best_match["image"].subject.lower():
                passed = True
                correct_top1 += 1
                result_str = f"MATCHED: '{best_match['image'].subject}' (sim: {best_match['sim']:.3f})"
            else:
                top_name = best_match["image"].subject if best_match else "NONE"
                result_str = f"MISMATCH: expected '{case['expected_subject']}', got '{top_name}'"
        else:
            if best_match is None:
                passed = True
                correct_top1 += 1
                top_raw = scored[0]
                result_str = f"SAFELY REJECTED (Reason: {top_raw['decision'].reason})"
            else:
                result_str = f"FAILED REJECTION: falsely approved '{best_match['image'].subject}'"

        print(f"Case #{case['id']:02d}: {case['title'][:40]}... -> {result_str}")

    precision = (correct_top1 / total_cases) * 100.0
    print("\n============================================================")
    print(f"Top-1 Precision Score: {correct_top1}/{total_cases} ({precision:.1f}%)")
    print("============================================================\n")

    db.close()

if __name__ == "__main__":
    run_evals()
