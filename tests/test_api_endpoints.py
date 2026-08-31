from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_post_creation_and_matching():
    # 1. Create a blog post about foxes
    post_payload = {
        "title": "Habitats and Hunting Traits of the Red Fox",
        "content": "Red foxes (Vulpes vulpes) hunt mice in open meadows using sharp hearing.",
        "target_subject": "red fox"
    }
    create_res = client.post("/posts", json=post_payload)
    assert create_res.status_code == 200
    post_id = create_res.json()["id"]

    # 2. Query recommended images
    match_res = client.get(f"/posts/{post_id}/images")
    assert match_res.status_code == 200
    data = match_res.json()
    assert data["post_id"] == post_id
    assert data["best_match"] is not None
    assert "fox" in data["best_match"]["subject"].lower()
    assert data["best_match"]["is_safe"] is True

def test_editorial_review_endpoint():
    payload = {"decision": "APPROVED", "reviewer_notes": "High resolution match"}
    res = client.post("/review/1", json=payload)
    assert res.status_code == 200
    assert res.json()["editorial_decision"] == "APPROVED"

def test_cost_metrics_endpoint():
    res = client.get("/metrics/costs")
    assert res.status_code == 200
    data = res.json()
    assert "total_calls" in data
    assert "estimated_total_usd" in data
