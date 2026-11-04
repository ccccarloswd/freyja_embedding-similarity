# Data in this repository

Only derived data is included: profiles, embeddings, selections and results. The raw data lakes, the TaBERT checkpoint and large intermediate files are excluded. All excluded files can be regenerated with the scripts in `src/` (see the main [README](../README.md#9-reproducing-the-experiments)).

| Path | Content | Produced by |
|---|---|---|
| `perfiles/freyjabm/` | FREYJA profiles of FREYJABM (680 text columns): raw, z-score normalised (`.csv`) and FREYJA's own pickle | FREYJA API |
| `perfiles/opendata/` | FREYJA profiles of the OpenData selection (5,241 text columns, 444 tables) | FREYJA API |
| `embeddings/freyjabm_warpgate/` | WarpGate embeddings (1,024 dims) of FREYJABM in four variants: `original`, `filas_azar_1`, `filas_azar_2` (three random rows) and `nombre_fijo` (fixed column name). `columnas.csv` maps rows to columns | `src/embeddings_warpgate.py` |
| `embeddings/freyjabm_mpnet/` | MPNet embeddings (768 dims) of FREYJABM: `original.npy` (unit length) and `sin_normalizar.npy` (before normalisation) | `src/embeddings_mpnet.py` |
| `embeddings/opendata_warpgate/`, `embeddings/opendata_mpnet/` | The same for the OpenData selection | idem |
| `seleccion_opendata.csv` | The 445 OpenData tables selected (one per family), with country and family | `src/seleccion_opendata.py` |
| `experimento/predicciones_freyjabm_*` | Cross-validation predictions of phases 1 and 2 (one value per pair, empty for pairs that are never tested) | `src/experimento.py` |
| `experimento/importancias_freyjabm_*` | Tree and permutation importances per fold | `src/experimento.py` |
| `experimento/modelos_freyjabm_*` | Preliminary model comparison on FREYJABM, per fold | `src/comparar_modelos.py` |
| `criba/resultados_warpgate.csv`, `criba/resultados_mpnet.csv` | Model screening on OpenData: time, trees, R² (train, validation, test), Spearman, Precision@10 (tie-aware, recomputed from the saved test predictions) and MAE for each model and configuration | `src/criba.py` |
| `analisis/notas_por_consulta_*.csv` | Test scores of RandomForest B, LightGBM C, XGBoost C and FREYJA's model for each query column (Spearman, tie-aware Precision@10, error sums for R²) | `src/analisis.py` |
| `analisis/bootstrap_*.csv` | Paired bootstrap over test tables: score of each model and of each difference, with 95 % and Bonferroni-corrected intervals | `src/analisis.py` |
| `analisis/curvas_*.csv` | Learning curves: validation and training scores with 10, 25, 50 and 100 % of the training tables | `src/analisis.py curvas` |
| `objetivos/resultados.csv` | Models trained on the dot product and with 95 features (D37, D38): time, trees, R² (train, validation, test), Spearman and Precision@10, per embedding, target and model | `src/objetivos.py` |
| `objetivos/notas_*.csv`, `objetivos/bootstrap_*.csv` | Scores per query column and paired bootstrap comparisons: cosine model against dot-product model, dot product from its parts, 95 against 35 features, and the three models with 95 features | `src/objetivos.py` |
| `objetivos/norma.csv`, `objetivos/norma_importancias_*.csv` | Predicting each column's embedding norm from its own profile, with permutation importances | `src/objetivos.py <emb> norma` |
| `objetivos/curvas_*.csv` | Learning curves with 35 and 95 features (WarpGate) | `src/objetivos.py <emb> <target> curvas` |
| `informe/resumen.csv` | Summary metrics of phases 1 and 2 | `src/informe.py` |

## Excluded and how to regenerate

| File | Size | How to regenerate |
|---|---|---|
| `raw/` (data lakes) | about 3 GB | Download (see [docs/data.md](../docs/data.md)) |
| `modelos/tabert_large_k3/` | 720 MB | From FREYJA's WarpGate implementation |
| `experimento/distancias_freyjabm.csv` | 170 MB | `src/experimento.py` (computed and cached on first run) |
| `criba/pares_opendata.pkl` | 660 MB | `src/criba.py` (computed and cached on first run) |
| `criba/predicciones/` (test predictions) | 480 MB | `src/criba.py` |
| `objetivos/predicciones/` (test predictions) | 170 MB | `src/objetivos.py` |
| `inventario_lb_opendata.csv` | 173 MB | Quick inventory of the OpenData lake |

## Licences

FREYJABM belongs to the FREYJA project. LakeBench OpenData is distributed under CC BY 4.0 / CC BY-SA 4.0. The profiles and embeddings here are derived from those tables.
