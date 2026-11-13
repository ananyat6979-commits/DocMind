# Remaining research and engineering work

The [measured protocol and failures](EVALUATION.md) replace the earlier unmeasured feature roadmap. Public benchmarks, human-label subsets, attribution comparisons, ablations, injection cases, basic auth/async ingestion/ANN and local load measurements are now implemented and recorded.

No useful calibrated automatic coverage was demonstrated. The next research step is a larger independently labeled evaluation of claim segmentation, evidence attribution and context-grouped selective calibration, with a timed reviewer study and cost-matched controls. Pre-register a new holdout before changing thresholds or models.

The next operational steps are incremental persistent indexing, multi-process snapshot coordination, tenant authorization, distributed queues and quotas, full hosted-service load tests, retention automation and recovery drills. Those are not completed production claims. Keep any further work within available free compute and explicit quotas; no paid fallback is part of this plan.
