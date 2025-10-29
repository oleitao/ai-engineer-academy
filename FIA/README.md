Dataset

- Source (Kaggle): https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset
- Description: a set of YouTube comments annotated with sentiment, suitable for sentiment classification tasks, temporal analyses, and semantic exploration.

Summary (version included in this repository)
- Base file: `FIA/YoutubeCommentsDataSet.csv`
- Records: ~18,408 rows (excluding header)
- Main columns:
  - `Comment`: comment text (typically already in lowercase and with little punctuation)
  - `Sentiment`: sentiment label with three classes — `positive`, `neutral`, `negative`
- Approximate class distribution (in the included file):
  - positive: 11,432 (~62.1%)
  - neutral: 4,638 (~25.2%)
  - negative: 2,338 (~12.7%)
- Format: UTF‑8 CSV, comma‑separated, first line is the header.

Derived files in the project
- `FIA/YoutubeCommentsDataSet_with_dates.csv`: version with an additional `published_at` column (simulated ISO 8601 timestamps) used in temporal analyses when real dates are not available.
- `FIA/supervised-models/YoutubeCommentsDataSet_enriched.csv`: output from the supervised module with extra columns (predictions and heuristic flags).

How it is used in the modules
- Supervised (`FIA/supervised-models`): uses `Comment` and `Sentiment` to train a TF‑IDF + Logistic Regression classifier and to generate enrichments (sarcasm, toxicity, intent, escalation).
- Unsupervised (`FIA/unsupervised-models`): uses `Comment` for embeddings, semantic clustering, deduplication (SimHash), and anomaly detection.
- Generative (`FIA/generative-llm`): uses `Comment` (and optionally `Sentiment`) for summaries, zero/few‑shot classification, sentiment explanations, and rewrites.

Quality and limitations
- Class imbalance (predominance of `positive`); evaluate metrics carefully and consider balancing/weighting.
- Predominantly English language; modules include cleaning and stop‑word options, but adapting to PT may require extra adjustments.
- For reliable temporal analyses, prefer real dates; the simulated `published_at` column is for demonstrations only.

License and attribution
- See the Kaggle page for the dataset’s terms of use and license.
- Dataset credit: author “atifaliak” on Kaggle.

