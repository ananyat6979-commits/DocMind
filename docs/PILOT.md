# Pilot protocol: 2 to 3 real users, 2 weeks

The numbers in a pilot report must come from real users. This kit makes the measurements quick and defensible.

## Before you start (30 minutes)
1. Pick 2 or 3 people who genuinely read long documents. Tell them: it is a free pilot, text of their questions and retrieved passages is sent to Groq, no confidential documents.
2. Run one instance natively and give them a link. Set a password first: `$env:DOCMIND_APP_PASSWORD="..."`.
   - Same Wi-Fi: `streamlit run app.py --server.address 0.0.0.0` and share `http://<your-ip>:8501`.
   - Remote: a free Cloudflare quick tunnel, see docs/CLOUD.md section 3. No Docker needed.
   - The document library is shared by everyone using the instance.
3. Set a reviewer PIN: `$env:DOCMIND_REVIEWER_PIN="1234"` (PowerShell). You or a colleague is the reviewer.

## Baseline (so "hours saved" is not a guess)
Each user does 5 representative tasks by hand with a stopwatch, on documents they care about (find the answer and the page). Record minutes in `pilot/baseline_timing.csv` (copy the template). Then they do 5 comparable tasks with DocMind, timing until they are willing to rely on the answer, including checking the cited page. Fill `assisted_minutes`.

Use different but comparable questions for the two conditions, and alternate which goes first across users to reduce order effects.

## During the pilot
- Each user enters their name in the sidebar (this is how adoption is counted).
- They click Helpful or Not helpful, and may enter how long the task would take by hand.
- You review the queue at least every other day (Review tab).

## After 2 weeks
```powershell
python scripts/pilot_report.py --invited 3 --baseline pilot/baseline_timing.csv --tasks-per-week 10
python scripts/update_readme.py
```
`--tasks-per-week` is each user's own estimate of how often they do such a task. State it as an assumption.

## What you may truthfully say
Use only the numbers the script printed, with the sample size. Template:
"Piloted with N users over 2 weeks: X of N adopted it, M timed tasks, median/mean Y minutes saved per task (95% CI a to b), P% of answers rated helpful, Q% of answers routed to human review."
If the pilot has fewer than 15 timed tasks, say "small pilot". If you did not run the pilot, do not claim adoption or hours saved.
