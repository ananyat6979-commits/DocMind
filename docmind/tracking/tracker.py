"""
MLflow tracking for every query.

We log: question, answer, latency, number of sources, reasoning steps.
This lets you show a dashboard of system performance in your README,
and gives you numbers like "P50 latency 1.2s, mean faithfulness 0.84".
"""
import logging
import mlflow
from docmind.config import CONFIG

logger = logging.getLogger(__name__)


class DocMindTracker:
    def __init__(self):
        mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
        mlflow.set_experiment("docmind_queries")

    def log_query(
        self,
        question: str,
        answer: str,
        latency_ms: float,
        num_sources: int,
        num_reasoning_steps: int,
    ) -> None:
        """Log a single query run to MLflow."""
        try:
            with mlflow.start_run(run_name="query"):
                mlflow.log_param("question_length", len(question))
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("num_sources", num_sources)
                mlflow.log_metric("reasoning_steps", num_reasoning_steps)
                mlflow.log_metric("answer_length", len(answer))
        except Exception as exc:
            # Tracking failure should never crash the main pipeline
            logger.warning(f"MLflow logging failed: {exc}")

    def log_evaluation(self, metrics: dict) -> None:
        """Log RAGAS evaluation results."""
        try:
            with mlflow.start_run(run_name="evaluation"):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, float(value))
        except Exception as exc:
            logger.warning(f"MLflow eval logging failed: {exc}")
