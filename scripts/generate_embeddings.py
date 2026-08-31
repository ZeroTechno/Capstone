import json
from src.database import SessionLocal, ImageModel, init_db
from src.embedding import generate_embedding

def embed_all_images():
    init_db()
    db = SessionLocal()
    images = db.query(ImageModel).filter(ImageModel.status == "processed").all()

    print(f"Generating vector embeddings for {len(images)} images...")
    count = 0
    for img in images:
        text_to_embed = f"Subject: {img.subject}. Category: {img.category}. Caption: {img.caption}. Tags: {', '.join(img.attributes)}"
        emb = generate_embedding(text_to_embed)
        img.embedding_json = json.dumps(emb)
        count += 1
        if count % 10 == 0:
            print(f"Embedded {count}/{len(images)} images...")

    db.commit()
    db.close()
    print(f"Completed: {count} image embeddings stored in database.")

if __name__ == "__main__":
    embed_all_images()
