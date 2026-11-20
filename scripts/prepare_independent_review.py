"""Build a portable blinded second-reviewer packet without loading earlier labels."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

from scripts.prepare_own_answer_review import HTML, prepare

FIELDS = {
    "different_person": "Are you a different person from the first reviewer?",
    "prior_involvement": "Did you help write or confirm these questions or earlier labels?",
    "seen_prior_results": "Have you seen the earlier judgments, scores or result summaries?",
}
ATTESTATION_SCRIPT = """
function captureAttestation(){
 if(s.reviewer_attestation)return true;
 let values={};for(let key of ['different_person','prior_involvement','seen_prior_results']){
  let value=document.getElementById(key).value;
  if(!value){alert('Please answer each reviewer-context question honestly. Every option is allowed.');return false}
  values[key]=value;
 }
 s.reviewer_code='R2';s.reviewer_attestation={...values,recorded_at:new Date().toISOString()};persist();
 for(let key of Object.keys(values))document.getElementById(key).disabled=true;
 return true;
}
for(let key of ['different_person','prior_involvement','seen_prior_results']){
 let field=document.getElementById(key);
 if(s.reviewer_attestation){field.value=s.reviewer_attestation[key];field.disabled=true}
}
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def build(root, profile, sources_root, cap_seconds=600):
    if root.exists() or root.with_suffix(".zip").exists():
        raise FileExistsError("Second-reviewer packets are immutable")
    if not 60 <= cap_seconds <= 600:
        raise ValueError("Keep this review inside the reserved ten-minute ceiling")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        draft = Path(temporary)
        prepare(draft, profile, cap_seconds=cap_seconds)
        packet = json.loads((draft / "actual-answer-packet.private.json").read_text())
        links = json.loads((draft / "actual-answer-linkage.private.json").read_text())
    by_id = {link["id"]: link for link in links}
    random.Random(19).shuffle(packet["items"])
    sources_root = sources_root.resolve()
    assets = {}
    links = []
    for index, item in enumerate(packet["items"], 1):
        link = by_id[item["id"]]
        item["id"] = "second-answer-" + str(index)
        link["id"] = item["id"]
        links.append(link)
        for key, directory, extension in (
            ("href", "sources", ".pdf"),
            ("image", "pages", ".png"),
        ):
            relative, separator, fragment = item[key].partition("#")
            source = (sources_root / relative).resolve()
            source.relative_to(sources_root)
            if source.suffix.lower() != extension or not source.is_file():
                raise ValueError("Missing or unexpected private source asset")
            target = directory + "/" + source.name
            if target in assets and assets[target] != source:
                raise ValueError("Colliding source asset names")
            assets[target] = source
            item[key] = target + (separator + fragment if separator else "")
    if sum(path.stat().st_size for path in assets.values()) > 100_000_000:
        raise ValueError("Portable review asset bound exceeded")
    packet.pop("sha256")
    packet.update(
        reviewer_code="R2",
        order_seed=19,
        review_role="second",
        independence="Self-reported reviewer context; not assumed from file creation",
    )
    packet["sha256"] = sha(
        json.dumps(packet, sort_keys=True, ensure_ascii=True).encode()
    )
    fields = "<fieldset><legend>Reviewer context</legend>"
    for key, label in FIELDS.items():
        fields += f'<label>{label}<select id="{key}"><option value="">Choose</option><option value="yes">Yes</option><option value="no">No</option><option value="unsure">Unsure</option></select></label>'
    fields += "</fieldset>"
    html = HTML.replace(
        "__DATA__", json.dumps(packet, ensure_ascii=True).replace("<", r"\u003c")
    )
    html = html.replace("__CAP_MS__", str(cap_seconds * 1000)).replace(
        "__CAP_MINUTES__", str(cap_seconds // 60)
    )
    html = html.replace(
        "docmind-own-actual-answer-labels.json",
        "docmind-independent-answer-labels.json",
    )
    html = html.replace("schema:'own-actual-review-v1'", "schema:'second-review-v1'")
    html = html.replace(
        "<h1>Review actual document answers</h1>",
        "<h1>Second review: document answers</h1><p>Work independently. Do not consult earlier labels, result tables or another reviewer while rating. Honest prior involvement is allowed and will be recorded. No name or contact details are requested. Finishing fewer items is expected.</p>"
        + fields,
    )
    html = html.replace(
        "document.getElementById('start').onclick=()=>{sync();",
        ATTESTATION_SCRIPT
        + "document.getElementById('start').onclick=()=>{if(!captureAttestation())return;sync();",
    )
    html = html.replace(
        "document.getElementById('export').onclick=()=>{sync();",
        "document.getElementById('export').onclick=()=>{if(!captureAttestation())return;sync();",
    )
    root.mkdir()
    (root / "actual-answer-review.html").write_text(html, encoding="utf-8")
    for target, source in assets.items():
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    write(root / "actual-answer-packet.private.json", packet)
    write(root / "actual-answer-linkage.private.json", links)
    instructions = (
        "PRIVATE SECOND-REVIEWER PACKET\n\nExtract this archive completely, then open actual-answer-review.html in a browser.\n"
        "Use at most ten recorded running minutes. Pause for breaks. Completing fewer items is fine.\n"
        "Judge answer correctness against the linked complete PDF and claim support against the displayed evidence.\n"
        "Do not consult earlier labels or published result tables. Answer the reviewer-context questions honestly.\n"
        "Click Export private labels, then return only docmind-independent-answer-labels.json to the project owner.\n"
        "Keep the packet and export private. No login, payment, inference or internet connection is needed.\n\n"
        "Sources: original project operations guide and NIST SP 800-63B-4, a US-government publication.\n"
        "NIST source: https://doi.org/10.6028/NIST.SP.800-63B-4\n"
    )
    (root / "README.txt").write_text(instructions, encoding="utf-8")
    public_to_reviewer = ["actual-answer-review.html", "README.txt", *sorted(assets)]
    archive_path = root.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in public_to_reviewer:
            archive.write(root / name, name)
    manifest = {name: sha((root / name).read_bytes()) for name in public_to_reviewer}
    write(
        root / "handoff-manifest.private.json",
        {
            "files": manifest,
            "zip_sha256": sha(archive_path.read_bytes()),
            "packet_sha256": packet["sha256"],
        },
    )
    return {
        "answers": len(packet["items"]),
        "claims": sum(len(i["claims"]) for i in packet["items"]),
        "cap_seconds": cap_seconds,
        "packet_sha256": packet["sha256"],
        "human_labels_created": 0,
        "prior_labels_loaded": False,
        "handoff_files": len(public_to_reviewer),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("sources_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.profile, args.sources_root)))
