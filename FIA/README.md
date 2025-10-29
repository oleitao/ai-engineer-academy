FIA — Feedback Intelligence Analytics (YouTube Comments)
=========================================================

End‑to‑end analysis of YouTube comments combining four approaches:

- Supervised ML: train a local sentiment model and add heuristic flags (toxicity, sarcasm, intent, escalation).
- Unsupervised NLP: embed, cluster, deduplicate near‑duplicates, and detect anomalies.
- Temporal analysis: build sentiment time series and alert on negativity spikes; optional events overlay.
- Generative LLM: summaries by group/topic, zero/few‑shot classification, sentiment explanations, and support‑reply rewrites.

Repository structure
- `FIA/supervised-models` — scikit‑learn pipeline + heuristics. CLI: `analyze_comments.py`.
- `FIA/unsupervised-models` — embeddings + clustering + SimHash dedup + anomalies. CLI: `unsupervised_analyze_comments.py`.
- `FIA/temporal-analyses` — trends, alerts, cohorts, and charts. CLI: `temporal_analysis.py`.
- `FIA/generative-llm` — OpenAI‑based analyses with local JSONL cache. CLI: `generative_analyze_comments.py`.

Quickstart (per module)
1) Create a virtual environment (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2) Supervised (local, no network)
   ```bash
   pip install -r FIA/supervised-models/requirements.txt
   python FIA/supervised-models/analyze_comments.py \
     --csv FIA/YoutubeCommentsDataSet.csv --text-col Comment --label-col Sentiment
   # Output: FIA/supervised-models/YoutubeCommentsDataSet_enriched.csv
   ```

3) Unsupervised (local; TF‑IDF by default)
   ```bash
   pip install -r FIA/unsupervised-models/requirements.txt
   python FIA/unsupervised-models/unsupervised_analyze_comments.py \
     --csv FIA/YoutubeCommentsDataSet.csv --text-col Comment
   # Outputs in FIA/unsupervised-models/: *_enriched.csv, clusters_summary.csv, near_duplicates.csv, anomalies.csv
   ```

4) Temporal analyses (local)
   ```bash
   pip install -r FIA/temporal-analyses/requirements.txt
   # If your CSV has no timestamps, simulate for demo purposes:
   python FIA/temporal-analyses/temporal_analysis.py --simulate-start 2025-01-01 --resample W
   # Outputs in FIA/temporal-analyses/output/: time_series.csv, alerts.csv, *.png
   ```

5) Generative LLM (requires API key; supports offline heuristics)
   ```bash
   pip install -r FIA/generative-llm/requirements.txt
   export OPENAI_API_KEY=sk-...
   # Summaries by an existing column (e.g., Sentiment)
   python FIA/generative-llm/generative_analyze_comments.py summarize \
     --csv FIA/YoutubeCommentsDataSet.csv --text-col Comment --group-col Sentiment
   # Offline mode (no network): add --offline to use heuristics
   ```

Typical workflow
1) Start with the base dataset `FIA/YoutubeCommentsDataSet.csv`.
2) Run Supervised to enrich with `sentiment_pred`, `toxicity_score`, `sarcasm_flag`, `intent_pred`, `needs_escalation`.
3) Run Unsupervised to get clusters and remove duplicates; inspect `clusters_summary.csv`, `near_duplicates.csv`, `anomalies.csv`.
4) Run Temporal analyses on the original or enriched CSV to monitor trends and spikes. Prefer real timestamps when available.
5) Use Generative LLM for executive summaries, zero/few‑shot labels, explanations, and support‑reply rewrites (with caching and offline fallbacks).

Notes
- Language: dataset is mostly EN; heuristics include PT/EN keywords. Stop‑words and locale can be adapted.
- Offline: all modules work fully offline except `generative-llm` (which provides heuristics with `--offline`).
- Caches: unsupervised caches HF assets under `FIA/unsupervised-models/{cache,hf}`; generative uses `FIA/generative-llm/.cache.jsonl`.

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
