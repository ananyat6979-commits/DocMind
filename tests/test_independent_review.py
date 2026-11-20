"""Synthetic packet fixtures check blinding and portable assets, not research labels."""

import hashlib
import json
import zipfile

import pytest

from scripts.prepare_independent_review import build


def fixture(tmp_path):
    profile = tmp_path / "profile"
    (profile / "private").mkdir(parents=True)
    sources = tmp_path / "sources"
    (sources / "corpus").mkdir(parents=True)
    (sources / "nist-qa").mkdir()
    (sources / "corpus/source.pdf").write_bytes(b"synthetic pdf fixture")
    (sources / "nist-qa/page-1.png").write_bytes(b"synthetic image fixture")
    (sources / "prior-labels.json").write_text("PREVIOUS_RATING_MUST_STAY_PRIVATE")
    gold = [
        {
            "task_id": "q:pdf",
            "question": "How long?",
            "source_page": 1,
            "group": "nist",
            "source_filename": "source.pdf",
            "answerable": True,
            "answer": "REFERENCE_ANSWER_MUST_STAY_PRIVATE",
        }
    ]
    rows = [
        {
            "task_id": "q:pdf",
            "format": "pdf",
            "status": "completed",
            "answer": "Ten minutes.",
            "release_state": "held",
            "nli_score": 0.01,
            "evidence": [
                {"claim": "Ten minutes.", "text": "The interval is ten minutes."}
            ],
        }
    ]
    (profile / "private/gold.json").write_text(json.dumps(gold))
    (profile / "private/records.json").write_text(json.dumps(rows))
    return profile, sources


def test_second_review_bundle_hides_labels_and_preserves_inputs(tmp_path):
    profile, sources = fixture(tmp_path)
    root = tmp_path / "second"
    result = build(root, profile, sources)
    assert (
        result["human_labels_created"] == 0 and result["prior_labels_loaded"] is False
    )
    packet = json.loads((root / "actual-answer-packet.private.json").read_text())
    content = {key: value for key, value in packet.items() if key != "sha256"}
    assert (
        packet["sha256"]
        == hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=True).encode()
        ).hexdigest()
    )
    assert packet["cap_seconds"] == 600 and packet["reviewer_code"] == "R2"
    assert packet["items"][0]["answer"] == "Ten minutes."
    assert packet["items"][0]["href"] == "sources/source.pdf#page=1"
    with zipfile.ZipFile(root.with_suffix(".zip")) as archive:
        assert set(archive.namelist()) == {
            "actual-answer-review.html",
            "README.txt",
            "sources/source.pdf",
            "pages/page-1.png",
        }
        html = archive.read("actual-answer-review.html").decode()
    assert "PREVIOUS_RATING_MUST_STAY_PRIVATE" not in html
    assert "REFERENCE_ANSWER_MUST_STAY_PRIVATE" not in html
    assert "release_state" not in html and "nli_score" not in html
    assert "docmind-independent-answer-labels.json" in html
    assert "captureAttestation()" in html and "seen_prior_results" in html
    with pytest.raises(FileExistsError):
        build(root, profile, sources)


def test_packet_refuses_assets_outside_the_source_root(tmp_path):
    profile, sources = fixture(tmp_path)
    gold = json.loads((profile / "private/gold.json").read_text())
    (tmp_path / "outside.pdf").write_bytes(b"must not be copied")
    gold[0]["source_filename"] = "../../outside.pdf"
    (profile / "private/gold.json").write_text(json.dumps(gold))
    with pytest.raises(ValueError):
        build(tmp_path / "second", profile, sources)
    assert not (tmp_path / "second").exists()


def review_source(root, *, second=False, verdict="correct", support="supported"):
    packet = json.loads((root / "actual-answer-packet.private.json").read_text())
    links = json.loads((root / "actual-answer-linkage.private.json").read_text())
    labels = {
        "packet_sha256": packet["sha256"],
        "elapsed_ms": 1000,
        "records": [
            {
                "id": item["id"],
                "finished": True,
                "human_adjudicated": True,
                "answer_correctness": verdict,
                "claims": {c["id"]: support for c in item["claims"]},
            }
            for item in packet["items"]
        ],
    }
    if second:
        labels.update(
            schema="second-review-v1",
            reviewer_code="R2",
            reviewer_attestation={
                "different_person": "yes",
                "prior_involvement": "no",
                "seen_prior_results": "no",
            },
        )
    return packet, links, labels


def paired_sources(tmp_path):
    from scripts.prepare_own_answer_review import prepare

    profile, sources = fixture(tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    prepare(first, profile)
    second = tmp_path / "second"
    build(second, profile, sources)
    return review_source(first), review_source(second, second=True)


def test_agreement_does_not_invent_kappa_for_constant_labels(tmp_path):
    from scripts.analyze_independent_review import compare

    first, second = paired_sources(tmp_path)
    result = compare([first], second)
    assert result["paired_answers"] == 1
    assert result["answers"]["observed_agreement"] == 1
    assert result["answers"]["cohen_kappa"] is None
    assert result["answers"]["kappa_interval95"] is None
    assert result["population_false_release_risk"] is None


def test_changed_answer_cannot_inherit_reference_rating(tmp_path):
    from scripts.analyze_independent_review import compare

    first, second = paired_sources(tmp_path)
    packet, links, labels = second
    packet["items"][0]["answer"] = "Different answer."
    content = {key: value for key, value in packet.items() if key != "sha256"}
    packet["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    labels["packet_sha256"] = packet["sha256"]
    with pytest.raises(ValueError, match="Changed displayed inputs"):
        compare([first], second)
    item = packet["items"][0]
    links[0]["input_sha256"] = hashlib.sha256(
        json.dumps(
            {"answer": item["answer"], "claims": item["claims"]}, sort_keys=True
        ).encode()
    ).hexdigest()
    result = compare([first], second)
    assert (
        result["paired_answers"] == 0
        and result["completed_without_matching_reference"] == 1
    )


def test_same_person_ratings_cannot_be_called_independent(tmp_path):
    from scripts.analyze_independent_review import compare

    first, second = paired_sources(tmp_path)
    second[2]["reviewer_attestation"]["different_person"] = "no"
    result = compare([first], second)
    assert result["independence_status"].startswith("Independence not established")
    with pytest.raises(ValueError, match="Repeated reference"):
        compare([first, first], second)


def test_kappa_reference_value_and_cluster_interval():
    from scripts.analyze_independent_review import agreement, bootstrap

    pairs = [
        ("correct", "correct"),
        ("correct", "incorrect"),
        ("incorrect", "incorrect"),
        ("incorrect", "incorrect"),
    ]
    assert agreement(pairs)["cohen_kappa"] == 0.5
    clusters = [{"answers": [pair], "claims": [pair, pair]} for pair in pairs * 3]
    interval = bootstrap(clusters, "claims")
    assert interval["resamples"] == 2000 and interval["paired_answer_clusters"] == 12
    assert interval["kappa_interval95"] is not None
