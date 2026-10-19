# DocMind: Requirements (one page)

Status: v1. Targets are not results until measured in a documented pilot or evaluation run.

## Problem
People who must answer specific questions from long PDF or DOCX documents (policy, research, contracts, reports) spend most of the time locating the passage, and they must be able to defend each answer with a page reference. Generic chat tools answer fluently but cite loosely or invent sources.

## Primary user
Knowledge workers in small operations, research, and proposal teams who read documents of 20 to 300 pages, ask 5 to 20 factual questions per document, and must trace every answer to a page or section. Secondary user: a reviewer who verifies flagged answers.

## Workflow
1. Upload one or more documents (PDF, DOCX, TXT, MD). The library keeps them; the user picks the scope.
2. Ask a question in natural language.
3. DocMind retrieves (dense + BM25, fused, reranked), reasons with a ReAct loop, and answers with citations of the form `(Source: file, p.N)`.
4. Every citation is checked against what was retrieved. Invalid ones are removed and counted.
5. Answers are routed by policy: abstentions, uncited or low-confidence answers, and a random audit sample go to the **human review queue**. A reviewer approves, corrects, or rejects.
6. The user marks the answer helpful or not, and optionally how long the task would have taken by hand.
7. Reviewed answers export to the evaluation set, so quality is re-measured on real questions.

## Scope
In: single-tenant, English text documents with a text layer, up to 50 MB per file, a shared pilot library, and a Groq-hosted LLM.
Out (v1): scanned PDFs without OCR, identity-based multi-user authorization, images and charts, non-English, confidential data, and durable free-tier hosting.

## Success metrics
| Metric | Target | Measured by | Result |
|---|---|---|---|
| Adoption: invited pilot users who used it | at least 2 of 3 | `scripts/pilot_report.py` | [fill after pilot] |
| Repeat use: users active on 2+ days | at least 2 of 3 | `scripts/pilot_report.py` | [fill] |
| Hours saved per user per week | at least 1.0 | stopwatch baseline CSV | [fill] |
| Helpful rate | at least 70% | in-app feedback | [fill] |
| Faithfulness (RAGAS) | at least 0.80 | `scripts/evaluate.py` | see README |
| Citation validity (valid / cited) | at least 0.98 | `scripts/evaluate.py` | see README |
| Unsupported answers on unanswerable questions | at most 10% | `scripts/evaluate.py` | see README |
| Reviewer turnaround for flagged answers | under 1 working day | review table timestamps | [fill] |
| p95 end-to-end latency | under 20 s on free tier | events table | [fill] |

Targets are goals, not results. A target is never reported as achieved until the Result column is filled from a measured run.

## Constraints and risks
- Groq free plan: per-model caps on requests and tokens per minute and per day (see console.groq.com/docs/rate-limits). A ReAct run uses several thousand tokens, so latency includes rate-limit waits.
- Privacy: text of retrieved passages leaves the machine. Do not pilot with confidential documents.
- Small-sample pilot: with 2 to 3 users, report counts and ranges, never population claims.
