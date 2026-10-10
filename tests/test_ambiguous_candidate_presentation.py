from src.product_feedback import build_user_feedback


def graph() -> dict:
    nodes = [{"id": "decision", "type": "decision", "label": "Decision"}]
    edges = []
    for index in range(8):
        target = f"target_{index}"
        nodes.append({"id": target, "type": "task", "label": f"Target {index}"})
        edges.append(
            {
                "id": f"e{index}",
                "source": "decision",
                "target": target,
                "label": f"branch_{index}",
                "branch_label": f"Branch {index}",
            }
        )
    return {"nodes": nodes, "edges": edges}


def spec() -> dict:
    return {
        "validation": {
            "issues": [
                {
                    "kind": "ambiguous_edge_reference",
                    "reference": "this branch",
                    "location": "semantic_goals[0].objects[0]",
                    "candidates": [f"e{index}" for index in range(8)],
                    "message": "multiple edge candidates",
                }
            ]
        }
    }


def test_ambiguous_candidates_show_first_five_and_keep_full_debug_candidates() -> None:
    feedback = build_user_feedback("needs_clarification", spec(), graph())

    message = feedback["message"]
    candidates = feedback["clarification"]["candidates"][0]["candidates"]

    assert len(candidates) == 8
    assert "Branch 0" in message
    assert "Branch 4" in message
    assert "e5" not in message
    assert "e7" not in message
    assert "e0" not in message
    assert "e4" not in message
    assert "Debug details" not in message
    assert any("\u8fd8\u6709 3 \u4e2a\u5019\u9009" in line for line in message.splitlines())


def test_ambiguous_edge_candidates_are_user_readable() -> None:
    feedback = build_user_feedback("needs_clarification", spec(), graph())
    first = feedback["clarification"]["candidates"][0]["candidates"][0]

    assert first["type"] == "edge"
    assert "Decision -> Target 0" in first["label"]
    assert "Branch 0" in first["label"]
