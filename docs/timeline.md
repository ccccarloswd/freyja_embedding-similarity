# Timeline

*Author: Carlos Martínez Nievas, Universitat Politècnica de Catalunya (UPC).*

How the project evolved. Decision numbers refer to [methodology.md](methodology.md).

## October 2026

**8 October. Design.**
- First methodological discussion: research question, null hypothesis and principles.
- Split by columns (D7), ranking metrics per query (D8), profiles through FREYJA's API (D9).

**9 October. Sampling and modelling.**
- Pair sampling per query column (D10) and exclusion of same-table pairs (D11).
- Regression of the raw cosine, one model per embedding (D12).
- Model selection on validation, never on test (D13).

**10 October. First data.**
- Selection rule (D15), WarpGate as first embedding (D16), text columns only (D18).
- Inventory script. Downloaded FREYJABM, SANTOS Small, OmniMatch and TUS Small.
- Finding: the five lakes contain only about 243 original tables.

**11 October. Searching for real data.**
- Evaluated GitTables and LB-WebTable. Adopted GitTables (D19) and the inclusion criterion (D20).
- Downloaded the full GitTables (96 topics, 871,195 tables). First quality check: three problems found.

**12 October. Families, clean folder and split.**
- Families rebuilt using the source repository and headerless-table detection (D21).
- No content filtering (D22). Clean folder verified column by column (D23). Decisions D24 to D26.
- Validation scheme with an external test (D27).

**13 October. Understanding WarpGate.**
- Read the WarpGate and TaBERT code: the embedding uses the column name and the first three rows.
- Confirmed it empirically with controlled inputs on the real model.
- Phased plan after a meeting with my supervisors (D28): FREYJABM first.

**14 to 16 October. Phase 1 set-up.**
- Phase 1 configuration (D29). WarpGate embeddings of FREYJABM with row and name diagnostics.
- Main target with the real column name (D30).
- FREYJABM profiles through the API. Studied FREYJA's profiling and distance code. Pair construction (D31).

**17 October. Phase 1 results.**
- Phase 1 pipeline (`src/experimento.py`), built step by step: R² 0.321, Spearman 0.465.
- Feature importance and interpretation.
- Saved the idea of a synonym dictionary learned from the lake.

**18 October. Names and a second embedding.**
- Version 2 with name-similarity features (v2a syntactic, v2b with learned synonyms).
- Searched for more data lakes and for a second embedding. Downloaded LakeBench OpenData.
- Phase 2 with MPNet, using DeepJoin's serialisation without statistics (D32).
- Additional metrics and figures to share the results (`src/informe.py`).

**20 October. Error by similarity range.** The model regresses to the mean and fails at both ends.

**24 October. Next round.**
- Feedback from my supervisors and plan for the next round (D33).
- GitTables replaced by LakeBench OpenData (four countries). About 22 GB freed.
- OpenData selection (445 tables), profiles, and WarpGate and MPNet embeddings.
- Preliminary model comparison on FREYJABM. New environment `.venv-ml/` for the models.

**25 October. Model screening design.**
- Gathered the universe of regressors and chose the candidates with explicit criteria (D34).
- Screening design on OpenData (D35). CatBoost excluded (blocked by the operating system's application control).

**26 and 27 October. Screening with WarpGate.**
- Stage A (defaults). EBM stopped after 5 h 27 min without finishing.
- RealMLP finished in 2 h 47 min (Spearman 0.493). KNeighbors discarded (0.360).
- The best tree models tie at 0.53 to 0.54, which points to the profiles as the limit.
- Configurations B and C for the five tree models: best result RandomForest B (0.552).

## November 2026

**1 November. Screening with MPNet.**
- References and configurations A, B and C of the five tree models.
- The five models are within 0.004 of each other (best RandomForest B, 0.560).
- ExtraTrees moves from last (WarpGate) to almost first.
- The ceiling appears again, at about 0.56.

**2 November. Publication.** Repository published with English documentation and figures.

**3 November. Statistical analysis.**
- RandomForest B chosen on validation with rule D15 (D36). The test only describes.
- Found that Precision@10 depended on how ties were broken. Replaced by its expected value under random tie-breaking, in the analysis and in the screening results.
- Paired bootstrap over the 89 test tables with Bonferroni-corrected intervals: RandomForest B is never worse than LightGBM or XGBoost and better with MPNet. The gain over FREYJA's model is solid in R² and uneven in ranking.
- Learning curves with 10, 25, 50 and 100 % of the training tables: about +0.01 of Spearman per doubling of the data, with the three models rising in parallel.
