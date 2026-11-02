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
| `criba/resultados_warpgate.csv`, `criba/resultados_mpnet.csv` | Model screening on OpenData: time, trees, R² (train, validation, test), Spearman, Precision@10 and MAE for each model and configuration | `src/criba.py` |
| `informe/resumen.csv` | Summary metrics of phases 1 and 2 | `src/informe.py` |

## Excluded and how to regenerate

| File | Size | How to regenerate |
|---|---|---|
| `raw/` (data lakes) | about 3 GB | Download (see [docs/data.md](../docs/data.md)) |
| `modelos/tabert_large_k3/` | 720 MB | From FREYJA's WarpGate implementation |
| `experimento/distancias_freyjabm.csv` | 170 MB | `src/experimento.py` (computed and cached on first run) |
| `criba/pares_opendata.pkl` | 660 MB | `src/criba.py` (computed and cached on first run) |
| `criba/predicciones/` (test predictions) | 480 MB | `src/criba.py` |
| `inventario_lb_opendata.csv` | 173 MB | Quick inventory of the OpenData lake |

## Licences

FREYJABM belongs to the FREYJA project. LakeBench OpenData is distributed under CC BY 4.0 / CC BY-SA 4.0. The profiles and embeddings here are derived from those tables.
