import json
import os
import sys
from pathlib import Path
from src.database import SessionLocal, ImageModel, init_db
from src.vision import analyze_image_with_retry

CONFIDENCE_THRESHOLD = 0.70

def run_batch():
    init_db()
    db = SessionLocal()
    pending = db.query(ImageModel).filter(ImageModel.status == "pending").all()

    print(f"Starting batch processing for {len(pending)} pending images...")

    processed = 0
    flagged = 0
    failed = 0

    for item in pending:
        print(f"Processing {item.filename}...", end="", flush=True)
        try:
            metadata = analyze_image_with_retry(item.file_path)
            item.subject = metadata.subject.lower()
            item.category = metadata.category.lower()
            item.attributes_json = json.dumps(metadata.attributes)
            item.caption = metadata.caption
            item.confidence = metadata.confidence
            
            # Requirement: Low-confidence classifications flagged instead of accepted
            if metadata.confidence < CONFIDENCE_THRESHOLD:
                item.is_flagged = True
                flagged += 1
                print(f" [FLAGGED low-confidence: {metadata.confidence:.2f}]")
            else:
                item.is_flagged = False
                print(f" [OK: {metadata.subject} ({metadata.confidence:.2f})]")

            item.status = "processed"
            processed += 1
        except Exception as e:
            item.status = "failed"
            item.error_message = str(e)
            failed += 1
            print(f" [FAILED: {e}]")

        db.commit()

    db.close()
    print(f"\nBatch Summary: {processed} processed, {flagged} flagged for review, {failed} failed.")

if __name__ == "__main__":
    run_batch()
