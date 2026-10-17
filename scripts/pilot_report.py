"""
Pilot report: adoption and hours saved, computed from the events table and the
baseline timing CSV. Prints exactly what it measured and the sample size, so
the numbers you put on a resume are defensible.

  python scripts/pilot_report.py --invited 3 --baseline pilot/baseline_timing.csv --tasks-per-week 10

baseline_timing.csv columns: user,task_id,manual_minutes,assisted_minutes
  manual_minutes   stopwatch time to answer the task WITHOUT DocMind
  assisted_minutes stopwatch time to answer the same kind of task WITH DocMind (incl. checking the citations)
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.evaluation.metrics import bootstrap_ci  # noqa: E402
from docmind.review.store import ReviewStore  # noqa: E402
from docmind.telemetry.events import EventLog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--invited", type=int, required=True, help="number of pilot users you invited")
    ap.add_argument("--baseline", default="pilot/baseline_timing.csv")
    ap.add_argument("--tasks-per-week", type=float, help="how many such tasks a user does per week (their estimate)")
    args = ap.parse_args()

    ev = EventLog()
    queries = ev.rows("query")
    feedback = ev.rows("feedback")
    by_user = defaultdict(list)
    for q in queries:
        by_user[q["user"]].append(q)

    print("== ADOPTION ==")
    active = [u for u, qs in by_user.items() if u != "anonymous" and len(qs) >= 1]
    repeat = [u for u, qs in by_user.items() if len({int(q["ts"] // 86400) for q in qs}) >= 2]
    print(f"invited: {args.invited} | used it at least once: {len(active)} | used on 2+ different days: {len(repeat)}")
    print(f"adoption rate: {len(active) / args.invited:.0%} | repeat-use rate: {len(repeat) / args.invited:.0%}")
    print(f"queries total: {len(queries)} | per active user: {len(queries) / max(len(active), 1):.1f}")
    if feedback:
        helpful = sum(1 for f in feedback if f["payload"].get("helpful"))
        print(f"helpful rate: {helpful}/{len(feedback)} = {helpful / len(feedback):.0%}")

    print("\n== QUALITY IN THE WILD ==")
    if queries:
        ab = sum(1 for q in queries if q["payload"].get("abstained"))
        rv = sum(1 for q in queries if q["payload"].get("needs_review"))
        lats = sorted(q["payload"].get("latency_ms", 0) for q in queries)
        print(f"abstained: {ab}/{len(queries)} | sent to review: {rv}/{len(queries)}")
        print(f"latency p50: {lats[len(lats) // 2] / 1000:.1f}s | p95: {lats[int(len(lats) * 0.95) - 1] / 1000:.1f}s")
    print("review queue:", ReviewStore().stats())

    print("\n== HOURS SAVED (stopwatch baseline) ==")
    p = Path(args.baseline)
    if not p.exists():
        print(f"No {p}. Copy pilot/baseline_timing.template.csv and fill it from real stopwatch timings.")
        return
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("manual_minutes") and r.get("assisted_minutes")]
    if not rows:
        print("Baseline file has no filled rows yet.")
        return
    saved = [float(r["manual_minutes"]) - float(r["assisted_minutes"]) for r in rows]
    mean, lo, hi = bootstrap_ci(saved)
    users = {r["user"] for r in rows}
    print(f"tasks timed: {len(rows)} across {len(users)} users")
    print(f"mean manual: {st.mean(float(r['manual_minutes']) for r in rows):.1f} min | "
          f"mean assisted: {st.mean(float(r['assisted_minutes']) for r in rows):.1f} min")
    print(f"minutes saved per task: {mean:.1f} (95% CI {lo:.1f} to {hi:.1f})")
    if args.tasks_per_week:
        wk = mean * args.tasks_per_week / 60
        print(f"hours saved per user per week at {args.tasks_per_week:g} tasks/week: {wk:.1f} "
              f"(CI {lo * args.tasks_per_week / 60:.1f} to {hi * args.tasks_per_week / 60:.1f})")
    if len(rows) < 15:
        print("NOTE: fewer than 15 timed tasks. Report this as a small pilot, not a general result.")


if __name__ == "__main__":
    main()
