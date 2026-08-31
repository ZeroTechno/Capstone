from src.guard import MismatchGuard

guard = MismatchGuard(min_similarity_threshold=0.55, min_confidence_threshold=0.60)

def test_mismatch_guard_rejects_wolf_on_fox_post():
    decision = guard.evaluate(
        post_title="The Hunting Behavior of Red Foxes in North America",
        post_content="Red foxes (Vulpes vulpes) hunt small rodents by leaping high in the air.",
        target_subject="red fox",
        image_subject="gray wolf",
        image_category="animal",
        image_attributes=["gray coat", "predator", "pack"],
        image_confidence=0.95,
        image_is_flagged=False,
        similarity_score=0.82
    )
    assert decision.is_safe is False
    assert decision.status == "REJECTED"
    assert "Animal category mismatch: expected fox, detected wolf" in decision.reason
    print("PROBE 3 PASSED: Wolf successfully rejected for fox post with reason:", decision.reason)

def test_mismatch_guard_approves_valid_fox():
    decision = guard.evaluate(
        post_title="Vulpes vulpes Ecology and Habits",
        post_content="Wild fox species thrive in forest edges and urban borders.",
        target_subject="red fox",
        image_subject="red fox",
        image_category="animal",
        image_attributes=["orange fur", "bushy tail"],
        image_confidence=0.94,
        image_is_flagged=False,
        similarity_score=0.89
    )
    assert decision.is_safe is True
    assert decision.status == "APPROVED"
    print("PROBE 2 PASSED: Red Fox approved with reason:", decision.reason)

def test_mismatch_guard_rejects_low_similarity():
    decision = guard.evaluate(
        post_title="Quantum Computing Algorithms in Python",
        post_content="Qubits and superposition state transformations.",
        target_subject="technology",
        image_subject="red deer",
        image_category="animal",
        image_attributes=["antlers"],
        image_confidence=0.90,
        image_is_flagged=False,
        similarity_score=0.22
    )
    assert decision.is_safe is False
    assert decision.status == "REJECTED"
    print("PROBE 4 PASSED: Low similarity safely rejected:", decision.reason)

if __name__ == "__main__":
    test_mismatch_guard_rejects_wolf_on_fox_post()
    test_mismatch_guard_approves_valid_fox()
    test_mismatch_guard_rejects_low_similarity()
    print("\nAll Phase 3 Guard Tests Passed!")
