from src.product_feedback import extract_user_candidates


def test_readable_edge_labels_cover_labeled_and_unlabeled_edges() -> None:
    graph = {
        "nodes": [
            {"id": "decision", "type": "decision", "label": "Decision"},
            {"id": "report", "type": "task", "label": "Generate Report"},
            {"id": "publish", "type": "task", "label": "Publish Result"},
        ],
        "edges": [
            {"id": "e_success", "source": "decision", "target": "report", "label": "success"},
            {"id": "e_report_publish", "source": "report", "target": "publish", "label": ""},
        ],
    }
    spec = {
        "validation": {
            "issues": [
                {
                    "kind": "ambiguous_edge_reference",
                    "reference": "branch",
                    "candidates": ["e_success", "e_report_publish"],
                    "message": "multiple graph edges match reference",
                }
            ]
        }
    }

    candidates = extract_user_candidates(spec, graph)[0]["candidates"]
    labels = {row["id"]: row["label"] for row in candidates}

    assert labels["e_success"] == "success \u5206\u652f\uff1aDecision -> Generate Report"
    assert labels["e_report_publish"] == "\u8fde\u7ebf\uff1aGenerate Report -> Publish Result"
