# Measured HotpotQA retrieval results

HotpotQA validation distractor paragraph retrieval; ten contexts per question.
200 held-out questions, disjoint from 200 calibration questions. 95% bootstrap CIs over questions.

| Model / variant | Recall@5 | MRR@5 | nDCG@5 | All support@2 |
|---|---|---|---|---|
| BAAI/bge-small-en-v1.5\|headers=False\|dense | 0.938 [0.912, 0.960] | 0.935 [0.906, 0.959] | 0.884 [0.857, 0.908] | 0.520 [0.450, 0.585] |
| BAAI/bge-small-en-v1.5\|headers=False\|hybrid | 0.902 [0.870, 0.930] | 0.930 [0.900, 0.956] | 0.857 [0.828, 0.883] | 0.470 [0.400, 0.535] |
| BAAI/bge-small-en-v1.5\|headers=False\|hybrid_rerank_depth10 | 0.885 [0.853, 0.915] | 0.942 [0.915, 0.967] | 0.857 [0.828, 0.884] | 0.485 [0.415, 0.555] |
| BAAI/bge-small-en-v1.5\|headers=False\|hybrid_rerank_depth5 | 0.897 [0.865, 0.927] | 0.945 [0.920, 0.969] | 0.866 [0.837, 0.893] | 0.495 [0.425, 0.565] |
| BAAI/bge-small-en-v1.5\|headers=False\|sparse | 0.835 [0.797, 0.868] | 0.890 [0.854, 0.924] | 0.791 [0.756, 0.822] | 0.360 [0.295, 0.430] |
| BAAI/bge-small-en-v1.5\|headers=True\|dense | 0.943 [0.917, 0.965] | 0.935 [0.908, 0.959] | 0.894 [0.867, 0.917] | 0.575 [0.505, 0.640] |
| BAAI/bge-small-en-v1.5\|headers=True\|hybrid | 0.925 [0.897, 0.950] | 0.945 [0.918, 0.968] | 0.887 [0.859, 0.911] | 0.560 [0.490, 0.625] |
| BAAI/bge-small-en-v1.5\|headers=True\|hybrid_rerank_depth10 | 0.890 [0.858, 0.920] | 0.951 [0.923, 0.974] | 0.864 [0.837, 0.891] | 0.495 [0.425, 0.570] |
| BAAI/bge-small-en-v1.5\|headers=True\|hybrid_rerank_depth5 | 0.902 [0.873, 0.930] | 0.952 [0.926, 0.975] | 0.871 [0.843, 0.896] | 0.500 [0.430, 0.570] |
| BAAI/bge-small-en-v1.5\|headers=True\|sparse | 0.863 [0.825, 0.895] | 0.911 [0.878, 0.940] | 0.823 [0.789, 0.852] | 0.420 [0.350, 0.485] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=False\|dense | 0.838 [0.800, 0.873] | 0.872 [0.833, 0.907] | 0.782 [0.747, 0.814] | 0.345 [0.280, 0.410] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=False\|hybrid | 0.873 [0.838, 0.905] | 0.880 [0.842, 0.913] | 0.807 [0.774, 0.838] | 0.380 [0.315, 0.445] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=False\|hybrid_rerank_depth10 | 0.885 [0.853, 0.915] | 0.942 [0.915, 0.967] | 0.857 [0.828, 0.884] | 0.485 [0.415, 0.555] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=False\|hybrid_rerank_depth5 | 0.880 [0.848, 0.910] | 0.945 [0.919, 0.969] | 0.854 [0.824, 0.882] | 0.480 [0.410, 0.550] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=False\|sparse | 0.835 [0.797, 0.868] | 0.890 [0.854, 0.924] | 0.791 [0.756, 0.822] | 0.360 [0.295, 0.430] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=True\|dense | 0.833 [0.795, 0.868] | 0.885 [0.847, 0.918] | 0.787 [0.752, 0.820] | 0.360 [0.295, 0.425] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=True\|hybrid | 0.892 [0.860, 0.922] | 0.903 [0.867, 0.933] | 0.834 [0.802, 0.863] | 0.425 [0.355, 0.490] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=True\|hybrid_rerank_depth10 | 0.890 [0.858, 0.920] | 0.951 [0.923, 0.974] | 0.864 [0.837, 0.891] | 0.495 [0.425, 0.570] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=True\|hybrid_rerank_depth5 | 0.885 [0.853, 0.915] | 0.952 [0.926, 0.975] | 0.862 [0.833, 0.888] | 0.490 [0.420, 0.560] |
| sentence-transformers/all-MiniLM-L6-v2\|headers=True\|sparse | 0.863 [0.825, 0.895] | 0.911 [0.878, 0.940] | 0.823 [0.789, 0.852] | 0.420 [0.350, 0.485] |

This is a bounded-context retrieval benchmark. It does not measure generated-answer accuracy, full-wiki retrieval, human review time, or entailment agreement.

HotpotQA annotations and context are CC BY-SA 4.0. Source: https://github.com/hotpotqa/hotpot
