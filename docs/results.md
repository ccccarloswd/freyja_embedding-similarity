# Results

*Author: Carlos Martínez Nievas, Universitat Politècnica de Catalunya (UPC).*

All results of the project, in the order in which they were obtained, including intermediate and preliminary ones. Decision numbers (D1 to D35) refer to [methodology.md](methodology.md).

**Metrics used throughout.**
- **R²** against the mean of the training split: the share of the cosine's variance the model explains. 0 means "no better than predicting the mean".
- **Spearman per query column** (main metric, D15): for each query column with at least 5 candidates, the rank correlation between the real and the predicted cosine of its candidates, averaged over query columns.
- **Precision@10:** for each query column, the fraction of its 10 most similar candidates (by the embedding) that the model also places in its top 10. On OpenData it is the expected value when ties are broken at random (see section 11).
- **MAE:** mean absolute error. Not comparable across embeddings, whose cosine ranges differ.

## Contents

1. [How the WarpGate embedding is computed](#1-how-the-warpgate-embedding-is-computed)
2. [WarpGate diagnostics](#2-warpgate-diagnostics)
3. [FREYJA profiles and distances](#3-freyja-profiles-and-distances)
4. [Phase 1: FREYJABM with WarpGate](#4-phase-1-freyjabm-with-warpgate)
5. [What the phase 1 model learned](#5-what-the-phase-1-model-learned)
6. [Version 2: column-name similarity](#6-version-2-column-name-similarity)
7. [Phase 2: FREYJABM with MPNet](#7-phase-2-freyjabm-with-mpnet)
8. [Additional metrics and error by similarity range](#8-additional-metrics-and-error-by-similarity-range)
9. [Preliminary model comparison on FREYJABM](#9-preliminary-model-comparison-on-freyjabm)
10. [Model screening on LakeBench OpenData](#10-model-screening-on-lakebench-opendata)
11. [Statistical analysis of the screening](#11-statistical-analysis-of-the-screening)

## 1. How the WarpGate embedding is computed

Read in FREYJA's `1_create_embeddings.py` and in the TaBERT source code:

- Each column becomes a **single-column table**, without context: the embedding does not see the rest of the table.
- TaBERT receives the column name, the type (always `text`) and all non-null values, but the K3 model only uses **the first three rows** (`table.data[:sample_row_num]`). There is no sampling.
- Each value is truncated to 5 tokens.
- The final vector is the mean of the tokens of each cell, then the mean of the three rows. 1,024 dimensions.
- The script reads CSV files with `latin1` encoding and lets pandas guess the types.

**Confirmed with the real model.** The `tb_config.json` of `tabert_large_k3` contains `sample_row_num: 3` and `max_cell_len: 5`. I also ran FREYJA's `column_embedding` function, unmodified, on a `Country` column of 20 countries:

| Test | Result |
|---|---|
| Same input twice | Identical embedding |
| Replace everything from row 4 (by colours, and with more rows) | **Identical** |
| Keep only the first three rows | **Identical** |
| Change the text of a value beyond its 5th token | **Identical** |
| Change the 3rd value (Italy → Brazil) | Changes. Cosine with the original 0.997 |
| Put three other rows first | Changes. Cosine 0.992 |
| Reverse the order | Changes. Cosine 0.981 |
| Change only the name: Country → Nation | Cosine 0.835 |
| Change only the name: Country → Color | Cosine 0.880 |
| Change only the name: Country → c0 | Cosine 0.740 |
| Same name, colour values | Cosine 0.879 |
| Name Color and colour values (an entirely different column) | Cosine 0.756 |

What enters the model for each row is `[CLS] [SEP] country | text | poland [SEP]`.

**Readings** (a single column, indicative only):
- **The column name weighs as much as or more than the values.** Changing only the name moves the cosine more (0.74 to 0.88) than changing all the values (0.88).
- Two unrelated columns give a cosine of 0.76: the useful range of the cosine is narrow (anisotropy).
- Consequence: the target depends on the name and three values, while the profile describes the whole column and does not see the name. The information mismatch is large.

## 2. WarpGate diagnostics

`src/embeddings_warpgate.py` computes four variants for FREYJABM: `original`, `filas_azar_1` and `filas_azar_2` (three random rows) and `nombre_fijo` (the same name, "column", for every column). Over the 658 text columns and the 212,411 pairs of columns from different tables:

| Variant | Mean cosine | Std. dev. | 1st percentile | 99th percentile |
|---|---|---|---|---|
| original | 0.753 | 0.090 | 0.497 | 0.916 |
| fixed name | 0.836 | 0.102 | 0.533 | 0.969 |

Agreement between variants (cosine of the same pair, computed two ways):

| Comparison | Pearson | Spearman per query | Mean difference |
|---|---|---|---|
| Three random rows vs three other random rows | 0.920 | 0.895 | 0.024 |
| First three rows vs three random rows | 0.917 | 0.881 | 0.026 |
| Real name vs fixed name | 0.629 | 0.526 | 0.097 |

- **The three rows matter less than feared.** Changing which rows are used keeps the cosine almost unchanged. The ceiling for this reason is high: a Spearman per query of about 0.90 is the maximum achievable.
- **The name matters a lot.** With the same name for every column, the cosine of each pair resembles the original only weakly (Pearson 0.63, about 40 % of shared variance), and the order of the most similar columns changes a lot (0.53).
- **Unavoidable limit:** when two columns have almost identical values and only the name tells them apart (demonyms and languages), their profiles are almost identical and no profile-based model can separate them.

With all 1,316 columns (numeric included) the figures are similar: 0.934 between random rows and 0.531 between real and fixed name.

## 3. FREYJA profiles and distances

**Profiles.** Computed with the API (`/profiles/profile_datalake`, `varchar_only: true`) on the FREYJABM folder: 680 columns of 160 tables, in 14 seconds. All 680 have an embedding. FREYJA counts 680 text columns against the 658 of my inventory because DuckDB classifies types differently (it includes booleans, for instance).

**How FREYJA computes them** (read in `app/core/profiling/profiler.py`, `metrics.py` and `app/core/distances/compute_distances.py`):
- Each CSV is read with DuckDB (`read_csv_auto`, types inferred from 100 rows, rows with errors ignored).
- Before profiling, values are **lower-cased and stripped of spaces and line breaks**. The embedding, in contrast, receives the original values.
- **34 metrics** per column: cardinality, entropy, uniqueness, incompleteness, value frequencies (mean, min, max, standard deviation, interquartile range, octiles 1 to 7), value percentages (min, max, standard deviation) and constancy, most frequent words and their Soundex, word lengths (max, min, mean), number of words and its statistics, first and last word, empty and binary column flags.
- **Z-score normalisation** of scale metrics, with the mean and standard deviation of **the whole lake**.
- **Distance between two columns, per metric:**
  - numeric metrics: **signed difference** (query − candidate). It is asymmetric: (A, B) and (B, A) have opposite signs.
  - frequent words: containment (the share of the query's words present in the candidate).
  - first and last word: Levenshtein distance.
- **On top of the 34, FREYJA adds `name_dist`**, the Levenshtein distance between the two column names. So FREYJA's model does see the column name, as a string, not as a meaning.

**Implications.**
- The model has a syntactic signal of the name. Predicting well a target where the name weighs so much would still be a syntax → semantics argument, because Levenshtein does not know that `Country` and `Nation` mean the same thing.
- Normalisation uses statistics of the whole lake, test columns included. It does not use the target, so it leaks nothing about the cosine. It is a FREYJA design choice, kept for fidelity (D29).

## 4. Phase 1: FREYJABM with WarpGate

Script: `src/experimento.py v1 warpgate`. Configuration D29 and D31: 680 columns of 160 tables, 453,620 ordered pairs (226,810 distinct), 35 features, 5-fold cross-validation with 5 groups of 32 tables (seed 0), FREYJA's Gradient Boosting settings. Query columns in each test fold have between 90 and 121 candidates (median 121).

| Fold | MAE model | MAE mean | R² | Spearman per query |
|---|---|---|---|---|
| 0 | 0.062 | 0.078 | 0.31 | 0.33 |
| 1 | 0.055 | 0.064 | 0.28 | 0.45 |
| 2 | 0.052 | 0.062 | 0.26 | 0.44 |
| 3 | 0.054 | 0.071 | 0.39 | 0.50 |
| 4 | 0.072 | 0.094 | 0.37 | 0.60 |
| **Mean (std.)** | **0.059 (0.008)** | **0.074 (0.013)** | **0.321 (0.060)** | **0.465 (0.101)** |

- **There is signal, and it is stable.** In all 5 folds the model clearly beats the mean: it reduces the error by 20 % and explains about 32 % of the variance of the cosine.
- **Ranking is harder.** The Spearman per query (0.47) is about half the ceiling set by row choice (0.90), and it varies across folds (0.33 to 0.60): with 160 tables, the result depends on which tables fall in each fold.
- FREYJA's published model uses 22 features selected for its own target (join quality). Here all 35 are used, because that selection was made for another target.
- An earlier run of the same configuration gave exactly the same numbers, which confirms that the process is reproducible.

## 5. What the phase 1 model learned

Two measures, averaged over the 5 folds:
- *Trees*: how much each feature was used during training (sums to 1).
- *Permutation*: how much the test R² drops when the feature is shuffled. The more reliable one.

| Feature | What it measures | Trees | Permutation |
|---|---|---|---|
| `name_dist` | Literal similarity of the column names | 0.09 | **0.069** |
| `first_word` | Similarity of the alphabetically first value | 0.11 | **0.043** |
| `last_word` | Similarity of the alphabetically last value | 0.13 | **0.040** |
| `frequency_5qo` | A percentile of the value frequencies | 0.16 | 0.030 |
| `frequency_avg` | Mean value frequency | 0.09 | 0.027 |
| `val_pct_min` | Share of the least frequent value | 0.08 | 0.015 |
| `len_avg_word`, `len_min_word` | Length of the values | 0.03 | 0.006 each |
| The other 27 | | | ≤ 0.005 each |

Features that measure roughly the same thing (the frequency percentiles) share importance, and permutation tends to under-rate them individually.

**Interpretation.**
- The model relies most on **literal string similarity**: of the name and of the extreme values in alphabetical order. These are exactly the features that "see" some of what the embedding sees (name and values).
- Next come features describing **the shape of the frequency distribution**: whether a column is repetitive (categories) or almost unique (identifiers, free text).
- Most structural statistics (entropy, cardinality, number of words) barely contribute to this target.
- Consistent with the information mismatch: WarpGate sees the name and three values, while most of the profile describes properties of the whole column.
- **Consequence for context-aware profiles:** this embedding is computed on a single-column table. Adding context from other columns to the profile cannot help predict it, except indirectly.

## 6. Version 2: column-name similarity

Motivation: the name is what weighs most in WarpGate, and the profile only captures its letter-by-letter similarity (`name_dist`).

**Synonym dictionary** (`src/sinonimos.py`), built at the time from the GitTables selection (see [data.md](data.md)):
- 25,439 columns from tables unrelated to FREYJABM. **2,165 columns whose content matched 80 % or more of some FREYJABM column were removed** (copies, to avoid cheating). 23,274 remained.
- Two columns from different families are related if they share at least 5 values and at least 50 % of the values of the smaller one (values present in more than 200 columns are ignored). This gives 7,088 column pairs.
- Names are split into words (`CountryRegionCode` → country, region, code). The similarity between two words is how often they appear together in those pairs, normalised by their frequency (at least 2 occurrences, smoothing 5). This gives 1,631 word pairs.
- Sensible examples: `language`–`lang`, `date`–`dt`, `team`–`abbr`, `title`–`movie`, `symbol`–`ticker`. There is also noise (words from headerless tables that were not detected).
- **Coverage:** 98 of the 361 words in FREYJABM names are in the dictionary, and 338 of the 680 columns have at least one.

**New features** (`src/experimento.py`, argument `v2a` or `v2b`): for each word of a name, its best match in the other name, averaged in both directions.
- `nombre_palabras` (v2a, syntactic): two words match if they are equal or one is an abbreviation of the other (same first letter and letters in order: `ctry` → `country`).
- `nombre_sinonimos` (v2b): the same, plus the dictionary similarity.

| Version | R² | Spearman per query | MAE |
|---|---|---|---|
| v1 (profile + `name_dist`) | 0.321 (0.060) | 0.465 (0.101) | 0.059 |
| v2a (+ words and abbreviations) | 0.328 (0.061) | 0.473 (0.100) | 0.059 |
| v2b (+ learned synonyms) | **0.336 (0.064)** | **0.478 (0.102)** | 0.058 |

- The gain is **small but consistent**: v2b beats v1 in R² in all 5 folds, and in Spearman in 4 of 5 (tied in the other).
- About half of the R² gain (+0.015) comes from splitting names properly (v2a) and the other half from the learned synonyms.
- In v2b, `nombre_sinonimos` is the 6th most important feature (permutation 0.028).
- Limitations: a small and noisy dictionary, coverage of half the columns, a single lake, no statistical test. The GitTables data it was built from was later dropped (see [data.md](data.md)), so v2b would need the dictionary to be rebuilt.

## 7. Phase 2: FREYJABM with MPNet

Embeddings: `src/embeddings_mpnet.py`, 1,316 columns, 768 dimensions, about 6 minutes on CPU. Serialisation D32. Median of 346 tokens per text, and 49 % of the texts are cut at 384 tokens (MPNet then sees only the first few dozen distinct values).

Training identical to phase 1: same profiles, features, pairs, folds, model and metrics. Only the embedding changes.

Cosine distribution: mean 0.30, standard deviation 0.16, from −0.13 to 0.995. A much wider range than WarpGate (mean 0.75, standard deviation 0.09).

| Embedding | R² | Spearman per query |
|---|---|---|
| WarpGate (TaBERT-K3) | 0.321 (0.060) | 0.465 (0.101) |
| MPNet | 0.281 (0.052) | 0.388 (0.075) |

By fold (MPNet): R² 0.22, 0.35, 0.31, 0.27, 0.26. Spearman 0.29, 0.49, 0.43, 0.37, 0.36.

Most important features with MPNet (permutation): `words_cnt_avg` 0.126, `freq_word_soundex_containment` 0.040, `frequency_7qo` 0.030, `words_cnt_max` 0.028, `name_dist` 0.024, `words_cnt_sd` 0.019.

- Signal in all 5 folds (R² from 0.22 to 0.35), somewhat below WarpGate.
- The most important feature is the mean number of words per value: short codes versus sentences really change the text MPNet reads. The second compares content (whether the most frequent words of the two columns sound alike).
- **Conclusion of the two phases:** with two very different embeddings, profiles predict the cosine moderately (R² 0.28 to 0.32, Spearman 0.39 to 0.47). In each case the model relies on the profile features that describe what that embedding reads: the name and the extreme values with WarpGate, the shape and content of the values with MPNet.

## 8. Additional metrics and error by similarity range

`src/informe.py`. Mean (standard deviation) over the 5 folds:

| Metric | WarpGate | MPNet |
|---|---|---|
| R² | 0.321 (0.060) | 0.281 (0.052) |
| Spearman per query | 0.465 (0.101) | 0.388 (0.075) |
| Precision@10 | 0.306 (0.038) | 0.264 (0.041) |
| Precision@10 with a random order | 0.083 | 0.083 |
| Bias in the top range (10 % of pairs with the highest cosine, predicted − real) | −0.082 | −0.226 |

- The model recovers about 3 of the 10 most similar columns, 3 to 4 times more than chance.
- On the most similar pairs the model falls short: it tends to predict close to the mean. That is the region that matters most for join discovery (D10).

Signed error by decile of the real cosine (test pairs, from least to most similar):

| Decile | WarpGate | MPNet |
|---|---|---|
| 1 (least similar) | +0.137 | +0.176 |
| 5 | −0.003 | +0.023 |
| 10 (most similar) | −0.085 | −0.228 |

The model regresses to the mean: it over-estimates dissimilar pairs and under-estimates very similar ones. With WarpGate the error is larger at the low end, with MPNet at the high end.

## 9. Preliminary model comparison on FREYJABM

> Exploratory run, kept as orientation only. It used few configurations per model (2 or 3), with no learning curves and no check that each model was well trained. The proper comparison is the screening of section 10.

`src/comparar_modelos.py`: the same experiment as phase 1, changing only the model. Light hyperparameter tuning with nested validation: in each fold, one of the 4 training groups acts as validation (Spearman per query), and the chosen configuration is retrained on the 4 groups before evaluating on the test fold.

| Model | WarpGate R² | WarpGate Spearman | WarpGate P@10 | MPNet R² | MPNet Spearman | MPNet P@10 |
|---|---|---|---|---|---|---|
| Gradient Boosting (FREYJA settings) | 0.321 (0.060) | 0.465 (0.101) | 0.306 | 0.281 (0.052) | 0.388 (0.075) | 0.264 |
| Ridge (linear) | 0.043 (0.030) | 0.302 (0.032) | 0.258 | 0.053 (0.091) | 0.150 (0.041) | 0.225 |
| HistGradientBoosting | **0.348** (0.081) | 0.500 (0.092) | **0.333** | 0.333 (0.061) | 0.420 (0.063) | 0.298 |
| Random Forest | 0.309 (0.082) | 0.488 (0.093) | 0.322 | **0.359** (0.044) | **0.439** (0.061) | **0.303** |
| Extra Trees | 0.262 (0.103) | 0.456 (0.104) | 0.317 | 0.336 (0.041) | 0.430 (0.072) | 0.291 |
| XGBoost | 0.324 (0.085) | 0.486 (0.084) | 0.329 | 0.337 (0.062) | 0.422 (0.070) | 0.293 |
| LightGBM | 0.340 (0.082) | **0.502** (0.084) | **0.333** | 0.334 (0.052) | 0.419 (0.057) | 0.293 |

- **The linear model is useless** (R² ≈ 0.05): the relation between profiles and cosine is clearly non-linear.
- **Higher-capacity tree models improve somewhat on FREYJA's**, but the differences among them are smaller than the variation across folds.
- Caveat: comparing models by their test score and keeping the best slightly inflates that score. This is why the next step fixed a proper protocol (D34, D35).

## 10. Model screening on LakeBench OpenData

Design D35. 444 tables (one per family), single 60/20/20 split by table. Pairs: training 620,686, validation 151,508, test 1,550,792, with 1,277 query columns in test (about 1,200 candidates each). The pairs do not depend on the embedding, only the target does.

### WarpGate

| Model | Config | Time | Trees | R² train | R² val | Spearman val | R² test | Spearman test | P@10 test | MAE test |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean (baseline) | A | 0 s |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.061 |
| GradientBoosting (FREYJA settings) | A | 4.7 min |  | 0.311 | 0.283 | 0.486 | 0.329 | 0.563 | 0.114 | 0.049 |
| RandomForest | A | 14.1 min |  | 0.950 | 0.334 | 0.534 | 0.358 | 0.533 | 0.136 | 0.048 |
| RandomForest | B | 3.6 min |  | 0.827 | 0.354 | **0.552** | 0.391 | 0.573 | 0.139 | 0.046 |
| RandomForest | C | 2.6 min |  | 0.627 | 0.357 | 0.549 | 0.395 | 0.578 | 0.145 | 0.046 |
| ExtraTrees | A | 5.9 min |  | 0.993 | 0.285 | 0.501 | 0.359 | 0.543 | 0.147 | 0.048 |
| ExtraTrees | B | 1.1 min |  | 0.768 | 0.258 | 0.476 | 0.339 | 0.544 | 0.140 | 0.048 |
| ExtraTrees | C | 35 s |  | 0.495 | 0.255 | 0.461 | 0.335 | 0.551 | 0.134 | 0.049 |
| HistGradientBoosting | A | 1.5 min | 2,330 | 0.719 | 0.333 | 0.533 | 0.346 | 0.535 | 0.143 | 0.049 |
| HistGradientBoosting | B | 2.6 min | 5,000 | 0.653 | 0.342 | 0.536 | 0.364 | 0.547 | 0.140 | 0.048 |
| HistGradientBoosting | C | 2.4 min | 2,761 | 0.799 | 0.337 | 0.539 | 0.350 | 0.539 | 0.138 | 0.048 |
| XGBoost | A | 3 s | 67 | 0.520 | 0.342 | 0.522 | 0.367 | 0.542 | 0.136 | 0.048 |
| XGBoost | B | 13 s | 610 | 0.459 | 0.349 | 0.536 | 0.382 | 0.570 | 0.131 | 0.047 |
| XGBoost | C | 8 s | 228 | 0.553 | 0.362 | 0.546 | 0.389 | 0.568 | 0.138 | 0.047 |
| LightGBM | A | 11 s | 431 | 0.586 | 0.356 | 0.540 | 0.373 | 0.557 | 0.136 | 0.047 |
| LightGBM | B | 11 s | 450 | 0.467 | 0.346 | 0.535 | 0.378 | 0.568 | 0.133 | 0.047 |
| LightGBM | C | 12 s | 226 | 0.592 | 0.360 | 0.549 | 0.387 | 0.569 | 0.139 | 0.047 |
| RealMLP | A | 2.8 h |  | 0.355 | 0.310 | 0.493 | 0.319 | 0.550 | 0.111 | 0.049 |
| KNeighbors | A | 16.3 min |  | 0.993 | 0.129 | 0.360 | 0.208 | 0.400 | 0.120 | 0.054 |

### MPNet

| Model | Config | Time | Trees | R² train | R² val | Spearman val | R² test | Spearman test | P@10 test | MAE test |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean (baseline) | A | 0 s |  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.123 |
| GradientBoosting (FREYJA settings) | A | 4.6 min |  | 0.433 | 0.381 | 0.474 | 0.403 | 0.527 | 0.176 | 0.094 |
| RandomForest | A | 14.0 min |  | 0.971 | 0.429 | 0.524 | 0.458 | 0.552 | 0.143 | 0.087 |
| RandomForest | B | 3.7 min |  | 0.897 | 0.468 | **0.560** | 0.495 | 0.584 | 0.162 | 0.085 |
| RandomForest | C | 2.7 min |  | 0.764 | 0.458 | 0.551 | 0.490 | 0.577 | 0.156 | 0.085 |
| ExtraTrees | A | 5.7 min |  | 0.999 | 0.469 | 0.556 | 0.504 | 0.594 | 0.175 | 0.084 |
| ExtraTrees | B | 1.4 min |  | 0.840 | 0.441 | 0.537 | 0.480 | 0.586 | 0.175 | 0.086 |
| ExtraTrees | C | 43 s |  | 0.599 | 0.414 | 0.521 | 0.457 | 0.579 | 0.176 | 0.089 |
| HistGradientBoosting | A | 3.0 min | 5,000 | 0.841 | 0.445 | 0.550 | 0.455 | 0.553 | 0.155 | 0.088 |
| HistGradientBoosting | B | 2.9 min | 5,000 | 0.738 | 0.442 | 0.545 | 0.460 | 0.561 | 0.152 | 0.087 |
| HistGradientBoosting | C | 4.6 min | 5,000 | 0.896 | 0.458 | 0.559 | 0.466 | 0.563 | 0.153 | 0.087 |
| XGBoost | A | 4 s | 122 | 0.675 | 0.420 | 0.525 | 0.442 | 0.546 | 0.155 | 0.089 |
| XGBoost | B | 20 s | 692 | 0.582 | 0.419 | 0.517 | 0.462 | 0.561 | 0.162 | 0.088 |
| XGBoost | C | 37 s | 959 | 0.789 | 0.459 | 0.556 | 0.477 | 0.567 | 0.157 | 0.086 |
| LightGBM | A | 23 s | 879 | 0.727 | 0.440 | 0.542 | 0.460 | 0.560 | 0.152 | 0.088 |
| LightGBM | B | 18 s | 734 | 0.608 | 0.422 | 0.520 | 0.472 | 0.564 | 0.154 | 0.087 |
| LightGBM | C | 1.0 min | 836 | 0.778 | 0.459 | 0.556 | 0.478 | 0.571 | 0.157 | 0.086 |

Precision@10 is the tie-aware version of section 11, recomputed from the saved test predictions. "Trees" is the number of boosting iterations kept by early stopping. Test numbers are reported for completeness: no decision was based on them.

### Not run or not completed

- **EBM:** configuration A (default interactions) was stopped after 5 h 27 min without finishing. B and C have not been run.
- **CatBoost:** its native library is blocked by the operating system's application control on the machine used.
- **RealMLP B and C, KNeighbors B and C, and all three with MPNet:** skipped for cost (RealMLP takes about 3 hours per configuration) or because the model was clearly discarded (KNeighbors).

### Reading

1. **Shared ceiling.** With both embeddings, the best configuration of every tree model and RealMLP lands between 0.49 and 0.56 Spearman on validation. Algorithms with very different inductive biases stop at the same level, which points to the information in the profiles as the bottleneck.
2. **Small effect of hyperparameters.** Configurations B and C changed the validation Spearman by 0.01 to 0.02 at most.
3. **Rankings depend on the embedding.** ExtraTrees is the worst tree model with WarpGate (0.501) and almost tied for the best with MPNet (0.556). Only RandomForest B is first on validation with both, by 0.003 (WarpGate) and 0.001 (MPNet).
4. **Fitting behaviour.**
   - RandomForest memorises with defaults (train R² 0.95 to 0.97) and generalises better when constrained, while also training 4 to 5 times faster.
   - ExtraTrees gets worse when constrained: its random split thresholds need deep trees.
   - XGBoost and LightGBM improve with larger trees (C).
   - RealMLP has a tiny train-validation gap (0.05), which suggests under-capacity rather than over-fitting. Configuration C (a wider network) would test this.
   - HistGradientBoosting hit its 5,000-iteration cap without early stopping in 4 of 6 runs. It may be slightly under-trained, although configuration C with WarpGate, which did stop on its own, reached the same level.
5. **Against FREYJA's reference.** On the test split, the best models clearly improve R² (WarpGate 0.329 → 0.391, MPNet 0.403 → 0.495) and, with WarpGate, Precision@10 (0.114 → 0.140). The Spearman per query improves little on test with WarpGate (0.563 → 0.573), and with MPNet FREYJA's reference has the best Precision@10 (0.176 against 0.162). The new models predict the value of the cosine better, but they do not necessarily find the top-10 columns better. Section 11 checks which of these differences are reliable.
6. **Validation versus test.** The gain over the reference is much larger on validation (+0.07 Spearman) than on test with WarpGate (+0.01). Two plausible reasons: validation ranks 100 random candidates per column while test ranks all of them (about 1,200), and the chosen configurations are slightly optimistic on the data used to choose them. The planned paired tests will tell which differences are real.

## 11. Statistical analysis of the screening

`src/analisis.py`. Models analysed: the best configuration of the three leading models on validation (RandomForest B, LightGBM C, XGBoost C) and FREYJA's model as the reference.

### 11.1 Choice of the model (D36)

Rule D15 on the validation scores already saved by the screening: the highest Spearman per query wins, and within 0.01 the simplest wins, with Random Forest considered simpler than boosting. RandomForest B is the best on validation with both embeddings (0.552 and 0.560), so it is chosen. The test split takes no part in the choice and is used only to describe.

### 11.2 Scores per query column and per table

The test has 1,277 query columns in 89 tables (median of 9 query columns per table, maximum 220). Spearman and R² recomputed column by column match the screening to the fourth decimal.

**Ties in Precision@10.** The first version of the metric broke ties by data order. Ties at rank 10 with WarpGate occur in 62 queries for the real cosine (duplicated columns), in 76 for RandomForest B and in **538 (42 %) for FREYJA's model**, whose 100 trees of depth 3 produce few distinct values. The metric is now the **expected Precision@10 if ties are broken at random**: a candidate tied at the boundary counts in proportion to the remaining places (2 places for 4 tied candidates give 0.5 each), on both sides. A constant prediction then scores exactly the random rate (0.0083). With WarpGate all scores drop by about 0.007 to 0.010. With MPNet they barely change, and FREYJA's model keeps its 0.176, so its advantage there was not caused by ties. The screening results were recomputed with this version from the saved test predictions.

### 11.3 Paired bootstrap over test tables

The 89 test tables are resampled with replacement 20,000 times. In each repetition every model is evaluated on the same tables (a paired comparison), and each metric is recomputed exactly as the original score: the mean over all query columns of the chosen tables for Spearman and Precision@10, and the ratio of summed errors for R². Intervals of differences are reported at 95 % and corrected by Bonferroni for the 36 comparisons presented (6 pairs of models × 3 metrics × 2 embeddings, 99.86 % intervals). A difference is reliable if its corrected interval excludes 0.

| Comparison on test | WarpGate | MPNet |
|---|---|---|
| RandomForest B − LightGBM C, Spearman | +0.004 [−0.005, +0.014] | **+0.013 [+0.004, +0.019]** |
| RandomForest B − XGBoost C, Spearman | +0.005 [−0.011, +0.018] | **+0.017 [+0.009, +0.023]** |
| RandomForest B − LightGBM C, R² | +0.004 [−0.005, +0.013] | **+0.018 [+0.010, +0.025]** |
| RandomForest B − XGBoost C, R² | +0.002 [−0.007, +0.014] | **+0.018 [+0.011, +0.026]** |
| RandomForest B − LightGBM C, Precision@10 | +0.000 [−0.014, +0.013] | +0.005 [−0.012, +0.017] |
| RandomForest B − FREYJA, Spearman | +0.010 [−0.013, +0.034] | **+0.057 [+0.045, +0.074]** |
| RandomForest B − FREYJA, R² | **+0.062 [+0.032, +0.094]** | **+0.092 [+0.072, +0.111]** |
| RandomForest B − FREYJA, Precision@10 | **+0.025 [+0.001, +0.049]** | −0.014 [−0.034, +0.013] |

Brackets: Bonferroni-corrected intervals. LightGBM C and XGBoost C against FREYJA's model behave like RandomForest B (R² about +0.06 with WarpGate and +0.07 with MPNet, Precision@10 with WarpGate +0.024 and +0.025, all reliable). Between LightGBM and XGBoost there is no reliable difference.

- **RandomForest B is never worse** than the other two leading models, and with MPNet it is reliably better. The choice is confirmed.
- **Against FREYJA's model, the gain is solid on the value** (R² with both embeddings) **and uneven on the order**: reliable with MPNet, not demonstrated with WarpGate. With WarpGate the new models find more of the true top 10. With MPNet FREYJA's model seems better at the top 10 (it wins in 94 % of the repetitions), but the difference is not reliable.
- The interval of each model on its own is wide (about ±0.04 of Spearman), because the overall level depends on which tables are in the test: some are easy and others hard for every model. The paired differences are much narrower because that effect cancels out.
- This replaces the Wilcoxon test planned in D35: both answer the same question, and the bootstrap over tables also respects the dependence between columns of the same table.

### 11.4 Learning curves

The three leading models trained on 10, 25 and 50 % of the training tables (3 random draws of tables each), keeping the pairs whose query column belongs to those tables, so that every column keeps its 100 candidates and the number of pairs falls in the same proportion. Validation is always the same. The 100 % point is the screening run. Spearman per query on validation:

| | 10 % | 25 % | 50 % | 100 % |
|---|---|---|---|---|
| Training pairs | about 52,000 | about 158,000 | about 343,000 | 620,686 |
| WarpGate RandomForest B | 0.500 | 0.532 | 0.543 | 0.552 |
| WarpGate LightGBM C | 0.495 | 0.536 | 0.541 | 0.549 |
| WarpGate XGBoost C | 0.486 | 0.533 | 0.542 | 0.547 |
| MPNet RandomForest B | 0.518 | 0.534 | 0.544 | 0.560 |
| MPNet LightGBM C | 0.506 | 0.530 | 0.546 | 0.556 |
| MPNet XGBoost C | 0.511 | 0.530 | 0.547 | 0.556 |

- **The curves are not flat.** Doubling the data from 50 to 100 % still adds between +0.005 and +0.017.
- **The returns diminish.** Each doubling adds about 0.01. Reaching 0.65 would need 8 to 10 more doublings, several hundred times more tables. The ceiling is not a wall but very slow growth, so the realistic lever is better profiles.
- **No model is data-hungry.** The three rise in parallel.
- The 100 % point is slightly optimistic, because the configuration was chosen on this same validation split.
- RandomForest B memorises (training R² about 0.83 against 0.35 on validation with WarpGate), and the gap does not shrink with more data.
