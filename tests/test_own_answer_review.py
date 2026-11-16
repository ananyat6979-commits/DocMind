import json
from scripts.prepare_own_answer_review import prepare


def test_packet_contains_real_answers_but_hides_routing_and_automatic_scores(tmp_path):
    root = tmp_path / "review"
    root.mkdir()
    profile = tmp_path / "profile"
    (profile / "private").mkdir(parents=True)
    gold = [
        {
            "task_id": "q:pdf",
            "question": "How long?",
            "source_page": 16,
            "group": "nist",
            "source_filename": "nist.pdf",
            "answerable": True,
        }
    ]
    records = [
        {
            "task_id": "q:pdf",
            "format": "pdf",
            "status": "completed",
            "answer": "Ten minutes.",
            "release_state": "held",
            "hold_reasons": ["invalid_citation"],
            "nli": {"score": 0.01},
            "evidence": [
                {
                    "claim": "Ten minutes.",
                    "text": "It expires in ten minutes.",
                    "status": "literal",
                }
            ],
        }
    ]
    (profile / "private/gold.json").write_text(json.dumps(gold))
    (profile / "private/records.json").write_text(json.dumps(records))
    result = prepare(root, profile)
    assert (
        result["answers"] == 1
        and result["claims"] == 1
        and result["human_labels_created"] == 0
    )
    packet = (root / "actual-answer-packet.private.json").read_text()
    assert "Ten minutes." in packet and "It expires in ten minutes." in packet
    assert (
        "release_state" not in packet
        and "invalid_citation" not in packet
        and "score" not in packet
    )
    linkage = json.loads((root / "actual-answer-linkage.private.json").read_text())
    assert linkage[0]["release_state"] == "held"


def test_timed_partial_review_never_invents_labels_or_population_risk():
    from scripts.analyze_own_answer_review import summarize
    import pytest

    packet = {
        "sha256": "p",
        "cap_seconds": 1800,
        "items": [{"id": "a", "claims": [{"id": "c"}]}, {"id": "b", "claims": []}],
    }
    linkage = [
        {"id": "a", "release_state": "draft"},
        {"id": "b", "release_state": "held"},
    ]
    labels = {
        "packet_sha256": "p",
        "elapsed_ms": 1000,
        "records": [
            {
                "id": "a",
                "finished": True,
                "human_adjudicated": True,
                "answer_correctness": "incorrect",
                "claims": {"c": "unsupported"},
            }
        ],
    }
    result = summarize(packet, linkage, labels)
    assert (
        result["unfinished_or_missing"] == 1
        and result["incorrect_reviewed_drafts"] == 1
    )
    assert (
        result["false_release_risk"] == "insufficient evidence"
        and result["reviewed_subset_false_release_fraction"] is None
    )
    labels["records"][0]["claims"] = {}
    with pytest.raises(ValueError, match="claim label"):
        summarize(packet, linkage, labels)


def test_missing_quote_text_stays_missing_without_inventing_support(tmp_path):
    root = tmp_path / "review"
    root.mkdir()
    profile = tmp_path / "profile"
    (profile / "private").mkdir(parents=True)
    gold = [
        {
            "task_id": "q:pdf",
            "question": "What is the limit?",
            "source_page": 16,
            "group": "nist",
            "source_filename": "nist.pdf",
            "answerable": True,
        }
    ]
    records = [
        {
            "task_id": "q:pdf",
            "format": "pdf",
            "status": "completed",
            "answer": "Twelve.",
            "release_state": "held",
            "evidence": [{"claim": "Twelve.", "text": None, "status": "missing"}],
        }
    ]
    (profile / "private/gold.json").write_text(json.dumps(gold))
    (profile / "private/records.json").write_text(json.dumps(records))
    result = prepare(root, profile)
    packet = json.loads((root / "actual-answer-packet.private.json").read_text())
    assert result["claims"] == 1 and packet["items"][0]["claims"][0]["quote"] == ""
    assert result["human_labels_created"] == 0


def test_metric_sensitivity_requires_identical_reviewed_inputs():
    import hashlib
    import pytest
    from scripts.analyze_own_answer_review import metric_sensitivity

    # Synthetic test judgments exercise accounting, not research annotations.
    claims = [{"id": "c", "claim": "Ten.", "quote": "Ten."}]
    # Packet preparation uses stable sequential claim identities.
    claims[0]["id"] = "claim-1"
    packet = {
        "sha256": "p",
        "cap_seconds": 1800,
        "items": [{"id": "a", "answer": "Ten.", "claims": claims}],
    }
    digest = hashlib.sha256(
        json.dumps({"answer": "Ten.", "claims": claims}, sort_keys=True).encode()
    ).hexdigest()
    linkage = [
        {
            "id": "a",
            "task_id": "q:pdf",
            "release_state": "draft",
            "input_sha256": digest,
        }
    ]
    labels = {
        "packet_sha256": "p",
        "elapsed_ms": 1000,
        "records": [
            {
                "id": "a",
                "finished": True,
                "human_adjudicated": True,
                "answer_correctness": "incorrect",
                "claims": {"claim-1": "unsupported"},
            }
        ],
    }
    records = [
        {
            "task_id": "q:pdf",
            "answer": "Ten.",
            "exact_match": 0,
            "evidence": [
                {"claim": "Ten.", "text": "Ten.", "chunk_id": "c", "status": "literal"}
            ],
            "sources": [{"chunk_id": "c"}],
        }
    ]
    result = metric_sensitivity(packet, linkage, labels, records)
    assert result["claim_location_by_human_support"] == {
        "citation_found=True:quote_located=True:unsupported": 1
    }
    records[0]["answer"] = "Changed answer."
    with pytest.raises(ValueError, match="Changed answer/evidence"):
        metric_sensitivity(packet, linkage, labels, records)


def test_new_packet_cap_and_source_links_are_frozen_with_inputs(tmp_path):
    root = tmp_path / "review"
    root.mkdir()
    profile = tmp_path / "profile"
    (profile / "private").mkdir(parents=True)
    gold = [
        {
            "task_id": "q:pdf",
            "question": "How long?",
            "source_page": 1,
            "group": "nist",
            "source_filename": "nist.pdf",
            "answerable": True,
        }
    ]
    records = [
        {
            "task_id": "q:pdf",
            "format": "pdf",
            "status": "completed",
            "answer": "Ten minutes.",
            "release_state": "draft",
            "evidence": [],
        }
    ]
    (profile / "private/gold.json").write_text(json.dumps(gold))
    (profile / "private/records.json").write_text(json.dumps(records))
    prepare(
        root,
        profile,
        cap_seconds=600,
        source_prefix="../core/",
        export_name="docmind-quality-answer-labels.json",
    )
    packet = json.loads((root / "actual-answer-packet.private.json").read_text())
    html = (root / "actual-answer-review.html").read_text()
    assert packet["cap_seconds"] == 600 and packet["items"][0]["href"].startswith(
        "../core/corpus/"
    )
    assert (
        "__CAP_" not in html
        and "cap=600000" in html
        and "docmind-quality-answer-labels.json" in html
    )


def test_changed_current_inputs_cannot_inherit_an_old_correct_judgment():
    from scripts.analyze_own_answer_review import combine_reviews
    from scripts.replay_quality_policy import fingerprint

    packet = {
        "sha256": "p",
        "cap_seconds": 600,
        "items": [{"id": "a", "question": "How long?", "answer": "Ten.", "claims": []}],
    }
    links = [
        {
            "id": "a",
            "task_id": "q:pdf",
            "release_state": "draft",
            "input_sha256": fingerprint("Ten.", []),
        }
    ]
    labels = {
        "packet_sha256": "p",
        "elapsed_ms": 1000,
        "records": [
            {
                "id": "a",
                "finished": True,
                "human_adjudicated": True,
                "answer_correctness": "correct",
                "claims": {},
            }
        ],
    }
    rows = [
        {
            "task_id": "q:pdf",
            "format": "pdf",
            "question": "How long?",
            "answer": "Eleven.",
            "evidence": [],
            "release_state": "draft",
        }
    ]
    result = combine_reviews(rows, [(packet, links, labels)])
    assert result["answer_correctness"] == {"unreviewed_identical_input": 1}
    assert result["false_release_fraction"] is None
