# Methodology and design decisions

*Author: Carlos Martínez Nievas, Universitat Politècnica de Catalunya (UPC).*

This document is the design log of the project. Every decision has a number, a rationale and a date, so that anyone can see not only what I did but why. Decisions are a guide, not a rulebook: when the context changed, I revised them and recorded the change (superseded decisions are kept and marked).

## 1. Objective

**Main question.** Given two columns A and B, to what extent do the differences between their profiles (purely syntactic statistics) predict the similarity between their embeddings?

    f( P(A), P(B) )  ≈  cos( E(s(A)), E(s(B)) )

- `P`: the column profile (FREYJA's code).
- `s`: the serialisation of the column into text.
- `E`: an embedding model.

**Target claim.** For unseen columns, unseen data and several embedding models, profiles predict the cosine similarity (above all its **relative order**) clearly better than trivial baselines. This would be empirical evidence that a profile is a distinctive representation of a column and that profile differences reflect differences of domain.

**Null hypothesis (H0).** The devil's advocate position that has to be refuted: *profiles carry no information about the cosine beyond trivial baselines*. If the results do not allow rejecting it, the claim cannot be made.

## 2. Principles

- **Generalisation.** Results must hold in most contexts, not in one chosen setting.
- **No fine-tuning where possible** (FREYJA's philosophy): a model trained once and applied to new data without re-training. One model per embedding model is acceptable if needed.
- **Several embedding models**, so that the result is not a property of one specific model.
- **Relative order matters most**, not the absolute value of the cosine. Embedding anisotropy is therefore not a problem and embeddings are not re-centred.
- **Simple reproducibility.** Few files. Anyone should be able to reproduce everything and understand the decisions by reading as little as possible.
- **No information leakage** between training and test.
- **Follow FREYJA's choices** (profiles, normalisation, statistics) unless there is an argument to change them.

## 3. Decisions

| # | Decision | Rationale | Date |
|---|---|---|---|
| D1 | Profiles are computed with the code of the FREYJA repository (34 statistics) | Fidelity to the project. Profile comparison is already solved there | 2026-10-08 |
| D2 | The initial embedding models are the ones used by the FREYJA project | Continuity | 2026-10-08 |
| D3 | The serialisation is defined before anything else, after studying how FREYJA does it | The target (the cosine) depends directly on it | 2026-10-08 |
| D4 | Parameters and statistical tests follow the updated FREYJA repository | They are already defined and reviewed | 2026-10-08 |
| D5 | Near-duplicate columns are detected and kept together in the same partition | Synthetic benchmarks split tables and replicate columns, which would cause leakage | 2026-10-08 |
| D6 | Broad and heterogeneous data lakes are used | Generalisation | 2026-10-08 |
| D7 | **Split by columns (or datasets):** columns are divided into train and test first, and pairs are then formed only within each side. No test pair contains information seen in training | If pairs were split, the same column would appear on both sides and the model could memorise it instead of generalising | 2026-10-08 |
| D8 | Evaluation includes **ranking metrics per query column**, not only global metrics | A global correlation can be high even if the model cannot rank the candidates of a specific column | 2026-10-08 |
| D9 | Profiles are obtained **through FREYJA's API** (served with uvicorn). The FREYJA repository is not included in this repository | No mixing of repositories and no third-party work. The API guarantees the same computation as FREYJA | 2026-10-08 |
| D10 | **Pair sampling per query column:** each column is paired with a maximum number of randomly chosen candidates, without forcing similar pairs. The test uses the natural distribution, organised by query. Errors are also evaluated by cosine range. If a problem appears in the high range, training sampling will be revisited | Does not bias the profile → cosine relation and no column dominates (see section 4.1) | 2026-10-09 |
| D11 | **Pairs of columns from the same table are excluded** | Whoever creates a table has consistent habits (format, encoding), which can bring unrelated columns closer and contaminate the target | 2026-10-09 |
| D12 | **The model predicts the raw cosine (regression), with one model trained per embedding model** | Training on one embedding already biases the predictor towards it. If one trained on one embedding and tested on others, it would be impossible to tell whether the result is general or specific to that training. Generality is shown if the result repeats independently with each embedding | 2026-10-09 |
| D13 | **Choice of ML technique (replaces a pilot study):** all data is prepared and the three candidate models (Ridge, Random Forest, Gradient Boosting) are trained on the full experiment. **The winner is chosen on validation data, never on the final test.** Only the winner is then evaluated in depth on the test set | Training is cheap. It avoids the small-pilot trap (with little data another model may win). Choosing on the test would inflate the winner's score | 2026-10-09 |
| D14 | **The ML technique is chosen with a single embedding model.** Only the winner is then trained with the other embeddings | Keeps the experiment manageable. Accepted risk: the winner may depend on the embedding used for selection. That embedding is fixed before seeing results. *Later relaxed: the screening on OpenData (D35) was run with both embeddings and showed that rankings do change* | 2026-10-10 |
| D15 | **Rule to choose the winning technique** (on validation): (1) the highest **mean Spearman per query column** wins. (2) If the difference is below 0.01, the simplest one wins. (3) Check: the winner's MAE must not be clearly worse than the others' | Simple, easy to verify and consistent with relative order being what matters (D8). MAE checks that values are also right | 2026-10-10 |
| D16 | **The first embedding model is WarpGate**, in the FREYJA repository implementation (TaBERT). It is also the embedding used to choose the ML technique (D14) | Supervisors' recommendation. Fixed before seeing results | 2026-10-10 |
| D17 | *Superseded by D19.* Data lakes: FREYJABM, SANTOS Small, TUS Small and OmniMatch (D3L has no direct download, so the reserve lakes were activated). TUS Large and SANTOS Large discarded | Real lakes plus one synthetic. Large synthetic lakes contribute many copies and little diversity | 2026-10-10 |
| D18 | **Text columns only for now** (`varchar_only: true` in FREYJA's API). Numeric columns are a later extension | FREYJA's default behaviour, and text embedding models give little meaning to numbers | 2026-10-10 |
| D19 | *Superseded by D33.* Adopted corpora: full GitTables (main source of real data, all 96 topics) and FREYJABM as real data. SANTOS Small, TUS Small and OmniMatch as synthetic data with a limited contribution | The synthetic lakes add up to only about 83 original tables. GitTables contributes thousands of real, independent tables from another origin (GitHub). LB-WebTable has tables that are too small and values that are too short | 2026-10-11 |
| D20 | **Column inclusion criterion:** text, at least **50 non-null values** and at least **20 distinct values** | Below that, profile and embedding carry very little information and many columns are nearly identical (for instance "Yes/No"). A starting point, open to revision | 2026-10-11 |
| D21 | **Table families** (`src/familias.py`). Two tables are in the same family if (a) they come from the same original table (synthetic lakes, by file name), (b) they come from the same GitHub repository (GitTables, via `src/repos_gittables.py`), (c) they have exactly the same header, or (d) they share a column with exactly the same values. **At most 3 tables per family**, chosen at random with seed 0. Headerless tables are detected and their first row is treated as data. The family is the unit split between train and test (D7) | Prevents train and test from sharing near-identical tables and prevents a few huge families from dominating. The source repository is the most reliable criterion: a single repository contributes 248,822 tables | 2026-10-12 |
| D22 | **No content is excluded unless FREYJA's API excludes it** (codes, identifiers, free text, and so on stay) | Filtering by hand would bias the data. A real data lake contains such columns. Results can be analysed by column type afterwards | 2026-10-12 |
| D23 | **Clean folder** (`src/lake_limpio.py`): selected tables are copied in a single format (UTF-8, commas, quoted text, header `c0, c1, …`). Check: the number of distinct values per column according to FREYJA must match the inventory | FREYJA's API profiles a whole folder, so the copy is needed. A single format ensures that profile and embedding are computed on values read the same way | 2026-10-12 |
| D24 | **No detector of edited copies across repositories is built.** Instead, the number of test columns with a near-twin in training (cosine ≈ 1) is measured, and results are reported with and without them | Proportionality: the detector is a lot of work and not standard practice. The measurement is almost free and tells whether the problem exists | 2026-10-12 |
| D25 | **The minimum of 20 distinct values is kept** (D20). After training, the model will be tested separately on columns with 5 to 19 distinct values | Low-cardinality columns (sex, weekday, continent) carry little information, generate trivial pairs and are not FREYJA's use case. A declared scope limit | 2026-10-12 |
| D26 | **The limit of 3 tables per family is kept** (D21) | Raising it only adds trivial pairs between tables of the same repository | 2026-10-12 |
| D27 | **Validation scheme** (`src/particion.py`). GitTables families are split at random with seed 0 into **train (70 %), validation (15 %) and test (15 %)**. The other lakes form an **external test** never used for training. All decisions are taken on validation. Test and external test are used once, at the end | Standard ML practice: the final score must not come from data used to decide. The external test measures whether results hold on data from another origin. Leave-one-lake-out was discarded (96 % is GitTables) | 2026-10-12 |
| D28 | **Phased plan** (after a meeting with my supervisors). **Phase 1:** only FREYJA's benchmark (FREYJABM), a Gradient Boosting model and WarpGate embeddings. **Phase 2:** the same with another embedding model. **Then:** share the results and decide whether to scale up | First a small, fast version, then scale up. D13–D15 and D27 are kept for the large-scale phase | 2026-10-13 |
| D29 | **Phase 1 configuration.** (a) **5-fold cross-validation grouped by table.** (b) Gradient Boosting with **FREYJA's model settings** (100 trees, learning rate 0.05, depth 3, subsample 0.8, at least 10 samples per leaf), untouched. (c) Baseline: **always predict the mean**. (d) Metrics: MAE, R² and Spearman per query column. (e) **Fidelity to FREYJA:** original benchmark folder, every text column the API profiles (without the D20 filter), real column names and the same embedding computation | (a) With 160 tables a single split would depend heavily on chance. This way every table is tested once and stability is visible. (b) No tuning means no validation is needed and no suspicion of over-tuning. (c) The cosine moves in a narrow band: without a baseline a small error means nothing. R² incorporates it. (e) The results must be those FREYJA's authors would obtain with their code | 2026-10-14 |
| D30 | **Main target: the cosine with the real column name** (WarpGate embedding as is). The fixed-name diagnostic is not used as a target for now | The name contributes much of the semantic power (it distinguishes a column of demonyms from one of languages with similar values). If purely syntactic profiles predict a target so loaded with semantics, the result is stronger | 2026-10-15 |
| D31 | **Pair construction.** Input: FREYJA's 34 distances plus `name_dist` (35 features), computed with its own function. Each pair enters **in both orders**, (A, B) and (B, A). Only pairs of columns from different tables | Distances rather than full profiles: it is the hypothesis (similar profiles → similar embeddings), trees struggle to compare two features with each other, some metrics only make sense compared (frequent words), and it is FREYJA's design. Both orders: FREYJA's distances are signed, so the model learns that order does not change the cosine. In statistics each pair counts once | 2026-10-16 |
| D32 | **An embedding serialisation must not contain statistics that are also in the profile.** For MPNet, DeepJoin's serialisation is used without its statistics: `"<table> information. <column>: <v1>, <v2>, …"` | If the embedding text included numbers such as the count of distinct values or of words per value, the model could predict the cosine by copying information present on both sides, which says nothing about the relation between profile and semantics. That profile and embedding come from the same values is the experiment. That they share pre-computed numbers is not | 2026-10-18 |
| D33 | **Next round, after feedback from my supervisors.** (1) Try other models and tune hyperparameters a little. (2) Also predict the Euclidean distance and the dot product between embeddings, on **unnormalised** vectors (with unit vectors the three measures rank pairs identically). (3) More data: a selection of LakeBench OpenData, one table per family. Also: give more weight to highly similar pairs **in training only**, always evaluating on all pairs. (4) Improve the profiles, after a meeting | The methodology was considered correct and the results a sufficient baseline. Highly similar pairs are few (about 2 %) and the model under-predicts them (it regresses to the mean and fails at both ends) | 2026-10-24 |
| D34 | **Candidates for the model comparison.** `RandomForest`, `ExtraTrees`, `HistGradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`, `EBM` and `RealMLP`. Fixed references: predicting the mean, and `GradientBoosting` with FREYJA's settings. Quick test to discard with data: `KNeighbors`. If `RealMLP` works very well, other networks of the same family will be tried (for instance `TabM`). Meta-models (`Stacking`, `Voting`) postponed | Universe: the 55 scikit-learn regressors plus the models of TabArena and AutoGluon (section 4.4). Discarded: linear, robust and generalised linear and projection models (the relation is not linear: `Ridge` gave R² ≈ 0.05), kernels (`SVR`, Gaussian processes: they do not fit in memory with a million pairs), single trees (unstable), foundation models (they accept a few tens of thousands of examples and are slow at prediction, against FREYJA's philosophy) | 2026-10-25 |
| D35 | **Model screening on OpenData** (`src/criba.py`). Single split by table (60 % training, 20 % validation, 20 % test, seed 0). Training and validation: each column with up to 100 random candidates from other tables of its part, in both orders. Test: all pairs of different tables. **3 configurations per model** (A defaults, B constrained, C high-capacity), chosen on validation (Spearman per query, D15). The number of trees of XGBoost and LightGBM, and RealMLP's stopping point, are decided on validation. HistGB and EBM use their own internal random validation. Paired comparison on the same test, with Wilcoxon and bootstrap intervals over tables. Then in-depth tuning of the chosen model only. **CatBoost is excluded**: the operating system's application control policy blocks its native library on this machine | With 444 independent tables a single split is reliable and costs a fifth of 5 folds. The paired comparison on the same query columns compensates for not having several folds. Three configurations covering different regimes avoid discarding a model because it fell into a bad region of hyperparameters | 2026-10-25 |

**Addenda to D35 (2026-10-26 and 2026-10-27).** EBM with default interactions was stopped after 5 h 27 min without finishing (judging by the worker processes, the stage without interactions took about 80 minutes). After stage A, configurations B and C were run only for the five tree models, since KNeighbors was clearly discarded (Spearman 0.36) and RealMLP took 2 h 47 min per configuration. RealMLP C and EBM B remain optional.

## 4. Design analyses

### 4.1 Pair sampling

**Key principle.**
- Sampling according to the *target* (the cosine) changes the profile → cosine relation the model learns. It biases it.
- Sampling according to the *input* (column characteristics: type, dataset, lake) does not bias it. It only changes which regions of the space are more or less represented.

**Test and training are different decisions.**
- **Test** must reflect real use: for each query column, all candidates (or a uniform sample), that is, the natural distribution. This allows unbiased evaluation and ranking metrics per query.
- **Training** can be chosen differently, because evaluation happens on the test.

| Option for training | For | Against |
|---|---|---|
| A. Uniform random pairs (natural) | No bias. Matches real use | Dissimilar pairs dominate. The high range (the one that matters for top-k) is under-represented. The model may regress to the mean |
| B. Stratified by cosine | Covers the full range | Biases the profile → cosine relation. Correctable with inverse-probability weights, at the cost of variance |
| C. Per query column, with a cap on candidates | Mimics the ranking task. No column dominates | Within each query it is still the natural distribution |
| D. Stratified by input characteristics | Does not bias the profile → cosine relation | Does not guarantee coverage of the high-cosine range |
| E. All pairs | No sampling decisions | Cost: n columns give about n²/2 pairs |

**Decision (D10):** natural test organised by query, training with C (plus D if types or lakes need balancing), no stratification by cosine in the main experiment (B with correction weights only as an ablation), and errors always reported by cosine range.

**Nuance.** The natural distribution depends on the composition of each data lake. There is no universal natural distribution, which connects with generalisation across lakes.

### 4.2 Target noise

The cosine of a given pair varies somewhat depending on which values enter the serialisation. No model can predict that variation, so it sets a performance ceiling. For WarpGate I measured it directly by recomputing the embeddings with three other random rows: the Spearman per query between the two versions is 0.895, which is the maximum achievable because of row choice (see [results.md](results.md#2-warpgate-diagnostics)).

### 4.3 Choice of ML technique

**Context.**
- About 35 input values per pair (the profile differences) and one continuous output (the cosine).
- Millions of pairs. CPU only.
- Preference for no fine-tuning (FREYJA's philosophy).
- Interest in interpreting which statistics approximate semantics.

**What the model predicts:** the raw cosine, one model per embedding model (D12). Alternatives discarded as the main experiment:
- Predicting the rank or percentile of the cosine with a single common model: it would mix embeddings in training.
- Learning to rank: it changes the question from "predict the cosine" to "order like the cosine", and produces no cosine values, so prediction error could not be measured.

**Initial candidates and why.** They cover the three big families, each with a different bias-variance trade-off:
- **Linear (Ridge):** high bias, low variance. Measures whether complexity adds anything.
- **Bagging (Random Forest):** reduces variance by averaging independent trees. Almost no tuning. Its *out-of-bag* validation is **not** valid here, because pairs are not independent.
- **Boosting:** reduces bias by correcting errors in sequence. State of the art on tabular data and what FREYJA uses.

If the conclusions hold with all three, they do not depend on the technique.

**Alternatives not chosen, in detail:**
- **Lasso:** like Ridge but sets some coefficients to zero. With highly correlated features (minimum, mean and maximum lengths) it picks one almost at random, which gives unstable interpretations.
- **Polynomial regression:** degree 2 on 34 features already gives about 600 terms, and still captures interactions very rigidly.
- **kNN:** in 34 dimensions distances concentrate (curse of dimensionality), it depends heavily on feature scale, and predicting requires comparing with millions of pairs. *Later confirmed with data: Spearman 0.36 on OpenData.*
- **Single decision tree:** very unstable. Random Forest and boosting exist precisely to fix this.
- **Neural networks (MLP):** on tabular data with heterogeneous features, trees usually win (Grinsztajn et al., 2022) and networks need much tuning. *Revisited in D34 with RealMLP, a modern tabular network with strong defaults.*
- **Siamese networks:** they learn a representation of the profile. That is representation learning, which this experiment wants to avoid, and it would answer a different question.
- **SVR and Gaussian processes:** they work with a similarity matrix between all training pairs. With one million pairs that is about 10¹² values (about 8 TB). Gaussian processes also cost about n³.

### 4.4 Universe of models and selection criteria (D34)

**How the universe was gathered.** `sklearn.utils.all_estimators(type_filter="regressor")` (55 regressors in scikit-learn 1.6.1), plus the models in the benchmarks and libraries that systematically compare the most competitive tabular models: TabArena (NeurIPS 2025, [arXiv 2506.16791](https://arxiv.org/abs/2506.16791)), [AutoGluon](https://auto.gluon.ai/dev/api/autogluon.tabular.models.html) and OmniTabBench (2026).

**Criteria.** Fit for the problem (regression, about 35 numeric features, between 300,000 and one million pairs), scalability, FREYJA's philosophy (lightweight, CPU, fast at prediction, little tuning), some interpretability, ability to model non-linear relations, variety of families, maintained and reproducible implementation.

| Family | Models | Decision |
|---|---|---|
| Linear | LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, OMP, BayesianRidge, ARD, SGD | Discarded (non-linear problem) |
| Robust and generalised linear | Huber, TheilSen, RANSAC, Quantile, PassiveAggressive, Poisson, Gamma, Tweedie | Discarded |
| Projection | PLSRegression, PLSCanonical, CCA | Discarded |
| Neighbours | KNeighbors, RadiusNeighbors | Quick test with KNeighbors |
| Kernel | SVR, NuSVR, KernelRidge, GaussianProcess, LinearSVR | Discarded (memory) |
| Single trees | DecisionTree, ExtraTree | Discarded (unstable) |
| Averaging ensembles | RandomForest, ExtraTrees, Bagging | RandomForest and ExtraTrees |
| Boosting | GradientBoosting, HistGB, XGBoost, LightGBM, CatBoost, AdaBoost, NGBoost | GradientBoosting (reference), HistGB, XGBoost, LightGBM, CatBoost |
| Interpretable additive | EBM, pyGAM, RuleFit | EBM |
| Tabular neural networks | MLPRegressor, FastAI/Torch MLP, RealMLP, TabM, ModernNCA, TabR, FT-Transformer, TabNet, SAINT, NODE | RealMLP (extended if it works very well) |
| Foundation models | TabPFN v2/2.5, TabICL, TabDPT, Mitra | Discarded |
| Meta-models | Stacking, Voting, TransformedTarget, MultiOutput, RegressorChain, MultiTask | Postponed |
| Other | IsotonicRegression (single feature), DummyRegressor (baseline) | Dummy as baseline |

### 4.5 Screening configurations (D35)

| Model | A (defaults) | B (constrained) | C (high capacity) |
|---|---|---|---|
| RandomForest | 300 trees | `max_features=1/3`, `min_samples_leaf=5` | `max_features=0.5`, `min_samples_leaf=20`, `max_samples=0.5` |
| ExtraTrees | 300 trees | as RandomForest B | as RandomForest C |
| HistGradientBoosting | `max_iter=5000`, early stopping | learning rate 0.05, 15 leaves, `min_samples_leaf=50` | learning rate 0.05, 127 leaves, `l2_regularization=1` |
| XGBoost | 5,000 trees, early stopping (50 rounds) on validation | learning rate 0.05, depth 4, row and column subsample 0.8 | depth 8, `reg_lambda=1` |
| LightGBM | early stopping (50 rounds) on validation | learning rate 0.05, 15 leaves, `min_child_samples=50` | 127 leaves, subsample 0.8, column subsample 0.8, `reg_lambda=1` |
| EBM | defaults | `interactions=0` | `interactions=30` |
| RealMLP (`RealMLP_TD`) | defaults, validation set for model selection | `p_drop=0.3`, weight decay 0.02 | hidden sizes 512 × 3 |
| KNeighbors | k = 10, distance weights, standardised | k = 50 | k = 200 |

## 5. Ideas saved for later

- **A synonym dictionary of column names learned from the data lake itself.** The name is what weighs most in WarpGate, and the profile only captures its literal similarity (`name_dist`). Idea: relate two column names if their columns share a lot of content (`Country` and `Nation` both contain Spain, France…). Built once, it would serve pairs of columns that share no values but whose names are known synonyms. It is purely syntactic, without embeddings or external models, and cheap to query, in line with FREYJA. A first version was tested in phase 1 (v2b). Alternatives discarded for now: WordNet or ConceptNet (external semantic knowledge with poor coverage of abbreviations and technical names) and an embedding of names (that would be predicting one embedding with another).
- **Richer profiles:** more specific metrics and context from other columns. Note that context cannot help predict WarpGate, because it embeds each column as a single-column table.
- **Clustering experiment:** do the clusters of the embedding space and of the profile space coincide?
- **Numeric columns**, excluded so far by D18.
- **Ablation:** train only on the 3 literal-similarity features (`name_dist`, `first_word`, `last_word`) and only on the 32 structural ones, to measure what each kind contributes.
