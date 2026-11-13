# Public evaluation data card

`data/testset.json` contains 200 unchanged SQuAD2 development questions with original human answer spans and answerability labels. There are 100 answerable and 100 unanswerable items, unique paragraph contexts, and article-disjoint calibration and held-out groups. `verified` means the original public human annotations were retained and checked against the source, not that new independent human review took place.

Source: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), Rajpurkar, Jia and Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD* (2018). Data license: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The source URL, canonical JSON SHA256 and per-paragraph hashes are in `corpus.json`. The selected subset and annotations retain that license. Source titles are represented by hashes in the corpus manifest; original questions, answers and spans are unchanged.

Run `python scripts/materialize_public_corpus.py` to retrieve the official data, verify all source/paragraph hashes, compare every question and annotation, and write ignored UTF-8 paragraphs under `data/public/documents`. This avoids duplicating source prose in Git. No private user uploads are included. Do not edit this frozen set using review-loop corrections.

Additional experiments use HotpotQA distractor validation data, from [Yang et al., HotpotQA](https://github.com/hotpotqa/hotpot), CC BY-SA 4.0. Published rank records retain source-title hashes and human support labels. CLIMATE-FEVER is from [Diggelmann et al.](https://github.com/tdiggelm/climate-fever-dataset); upstream licensing is not specified in its dataset card, so only derived scores, label IDs, offsets and hashes are redistributed, not claim or evidence prose. Wikipedia evidence may be incomplete or disputed.

A second 200-item SQuAD2 subset and its hash-only corpus manifest are under `reports/benchmarks/critic`. Its held-out contexts exclude the earlier held-out contexts. Both studies have separate article-disjoint calibration and evaluation groups. These are development-data experiments with unknown model pretraining exposure, not private unseen benchmarks.
