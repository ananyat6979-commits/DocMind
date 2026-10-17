"""
MLflow tracking for evaluation runs (not per-query: per-query data lives in the
events table, which is cheaper and queryable with SQL).
mlflow is imported lazily; importing it costs seconds.
"""
import logging
from typing import Dict, Optional

from docmind.config import CONFIG

logger = logging.getLogger(__name__)


class DocMindTracker:
    def __init__(self, experiment: str = "docmind_eval"):
        self.experiment = experiment

    def _mlflow(self):
        import mlflow
        mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment)
        return mlflow

    def log_evaluation(self, metrics: Dict[str, float], params: Optional[Dict] = None,
                       artifact_paths: Optional[list] = None, run_name: str = "evaluation") -> None:
        try:
            mlflow = self._mlflow()
            with mlflow.start_run(run_name=run_name):
                for k, v in (params or {}).items():
                    mlflow.log_param(k, v)
                for k, v in metrics.items():
                    if v is not None:
                        mlflow.log_metric(k, float(v))
                for p in artifact_paths or []:
                    mlflow.log_artifact(str(p))
        except Exception as exc:
            logger.warning(f"MLflow logging failed (continuing): {exc}")
