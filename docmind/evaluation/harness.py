"""
RAGAS evaluation harness.

RAGAS (Retrieval Augmented Generation Assessment) measures four things:
  - faithfulness:        are the claims in the answer supported by the retrieved context?
  - answer_relevancy:    does the answer actually address the question asked?
  - context_precision:   are the top-ranked contexts actually relevant to the question?
  - context_recall:      does the retrieved context cover the ground-truth answer?

For context_recall you need ground truth answers (testset).
For the other three, you just need questions : RAGAS uses the LLM as a judge.

The eval set lives in a JSON file: a list of {question, ground_truth} dicts.
You build this once from representative questions about your test documents.
On Kaggle you can run this against a large testset without laptop RAM limits.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from docmind.agent.react import ReActAgent
from docmind.tracking.tracker import DocMindTracker

logger = logging.getLogger(__name__)


def load_testset(path: str) -> List[Dict]:
    """
    Load evaluation testset from JSON.
    Format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    testset_path: str,
    agent: Optional[ReActAgent] = None,
    output_path: str = "evaluation_results.json",
) -> Dict[str, float]:
    """
    Run RAGAS evaluation over a testset.

    Steps:
      1. Load testset (question + ground_truth pairs).
      2. Run agent.run(question) for each question.
      3. Collect (question, answer, contexts, ground_truth).
      4. Pass to RAGAS evaluate().
      5. Log results to MLflow and save to JSON.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    agent = agent or ReActAgent()
    testset = load_testset(testset_path)
    tracker = DocMindTracker()

    questions, answers, contexts, ground_truths = [], [], [], []

    logger.info(f"Running evaluation on {len(testset)} questions...")

    for item in testset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        try:
            response = agent.run(question)
            # RAGAS expects contexts as a list of strings per question
            context_texts = [s.chunk.content for s in response.sources]

            questions.append(question)
            answers.append(response.answer)
            contexts.append(context_texts)
            ground_truths.append(ground_truth)

            logger.info(f"✓ '{question[:60]}...'")

        except Exception as exc:
            logger.warning(f"Skipping question due to error: {exc}")

    # Build HuggingFace Dataset (RAGAS expects this format)
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS — uses Groq as judge LLM internally
    metrics_to_run = [faithfulness, answer_relevancy, context_precision]
    if all(gt for gt in ground_truths):
        metrics_to_run.append(context_recall)

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics_to_run,
    )

    metrics_dict = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
    }
    if context_recall in metrics_to_run:
        metrics_dict["context_recall"] = result["context_recall"]

    # Log and save
    tracker.log_evaluation(metrics_dict)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Evaluation complete. Results: {metrics_dict}")
    return metrics_dict