import os
from pathlib import Path
from PIL import Image, ImageDraw
from src.database import init_db, SessionLocal, ImageModel

IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    ("red_fox", 10, "Red Fox (Vulpes vulpes) with distinctive orange fur"),
    ("gray_wolf", 10, "Gray Wolf (Canis lupus) predator of the northern forests"),
    ("domestic_dog", 10, "Golden Retriever domestic dog playing outdoors"),
    ("grizzly_bear", 10, "Grizzly Bear fishing along a shallow river"),
    ("red_deer", 8, "Red Deer stag with antlers standing in a meadow"),
    ("blurry_animal", 2, "Indistinct blurry shape in dense fog"),
]

def generate_sample_images():
    init_db()
    db = SessionLocal()
    
    total_created = 0
    for name, count, desc in CATEGORIES:
        for idx in range(1, count + 1):
            filename = f"{name}_{idx:02d}.jpg"
            filepath = IMAGE_DIR / filename
            
            if not filepath.exists():
                img = Image.new("RGB", (640, 480), color=(70 + (idx*10)%100, 90 + (idx*8)%80, 110 + (idx*5)%90))
                draw = ImageDraw.Draw(img)
                draw.rectangle([(20, 20), (620, 460)], outline="white", width=4)
                draw.text((40, 200), f"Sample: {name} #{idx}", fill="white")
                draw.text((40, 240), desc, fill="yellow")
                img.save(filepath, "JPEG")

            existing = db.query(ImageModel).filter(ImageModel.filename == filename).first()
            if not existing:
                record = ImageModel(
                    filename=filename,
                    file_path=str(filepath),
                    status="pending"
                )
                db.add(record)
                total_created += 1

    db.commit()
    db.close()
    print(f"Seed completed: ~50 images prepared in {IMAGE_DIR}")

if __name__ == "__main__":
    generate_sample_images()
