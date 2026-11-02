# Data

*Author: Carlos Martínez Nievas, Universitat Politècnica de Catalunya (UPC).*

This document describes every data source I evaluated, how the final selections were built and the risks I identified. Raw data lakes are not included in the repository (they are third-party data and too large). What is included is listed in [`data/README.md`](../data/README.md).

## 1. Data lakes in use

| Data lake | What it is | Source | Used for |
|---|---|---|---|
| **FREYJABM** | FREYJA's own benchmark: 160 real tables | FREYJA benchmarks (`datalake.zip`) | Phases 1 and 2 (5-fold cross-validation) |
| **LakeBench OpenData** | Real tables from open-data portals of Canada, the USA, the UK and Singapore (finance, trade, industry, energy, media…) | `opendata_sampled_1k.tar.gz` (673 MB, at most 1,000 rows per table) from [Zenodo 15499092](https://zenodo.org/records/15499092). Licence CC BY 4.0 / CC BY-SA 4.0 | Model screening (single 60/20/20 split) |

### 1.1 LakeBench OpenData: inventory

Extracted only the `datalake/` folder (2.8 GB uncompressed).

| | OpenData | FREYJABM |
|---|---|---|
| Tables | 7,970 (Canada 3,785, USA 2,359, UK 1,057, Singapore 769) | 160 |
| Text columns | 69,378 | 680 (profiled) |
| Included with D20 (≥ 50 non-null, ≥ 20 distinct) | **35,168 in 6,494 tables** | 352 |
| Distinct headers (tables with included columns) | 3,243 | |
| Rows per table (median) | 1,000 (cap of this version) | |

About 50 times more columns than FREYJABM and about 3,200 distinct schemas. Open-data portals publish many versions of the same table (by year, by region), so tables must be grouped into families before splitting.

### 1.2 LakeBench OpenData: selection

`src/seleccion_opendata.py` → `data/seleccion_opendata.csv`.

- **Families:** same header, or a text column with exactly the same values (only columns with ≥ 50 values and ≥ 20 distinct). 7,533 tables with text in 3,291 families.
- **One table per family**, families in random order (seed 0), until 5,000 text columns are reached.
- **Result: 445 tables, 5,003 text columns** by my inventory. By country: Canada 220, USA 150, Singapore 56, UK 19. The UK is small because its 1,057 tables form few families (many versions of the same tables).
- FREYJA's API profiled **5,241 text columns from 444 tables** (one table failed because of a duplicated column name, and DuckDB classifies types slightly differently from pandas).
- Embeddings: WarpGate 5,239 columns in 16 minutes (2 columns become empty after cleaning), MPNet 5,241 columns in 16 minutes (median 107 tokens, 37 % of texts truncated). The screening uses the columns that have both.

**Column names across tools.** WarpGate reads CSV files as `latin1` while FREYJA reads them as UTF-8 and cleans names (BOM, quotes, line breaks). To match each embedding to its profile, `nombre_como_freyja()` in the embedding scripts reproduces FREYJA's version of every name.

## 2. The road to the data: lakes evaluated and discarded

Finding enough real, diverse data was one of the hardest parts of the project. This section documents the path, because the reasons for each discard are part of the methodology.

### 2.1 First candidates (2026-10-10)

| Data lake | Source | Size |
|---|---|---|
| FREYJABM | FREYJA benchmarks | 33.7 MB |
| SANTOS Small | [Zenodo 7758091](https://zenodo.org/records/7758091) | 90.5 MB |
| OmniMatch (CG and CR) | [Zenodo 15705578](https://zenodo.org/records/15705578) | 98.1 MB |
| TUS Small | [Zenodo 15499092](https://zenodo.org/records/15499092) (a repackaged copy, the official TUS server is down) | 441.5 MB |
| D3L | No direct download source | |

Inventory (`src/inventario.py`). Type classification by pandas, indicative only:

| Lake | Tables | Original tables | Columns | Text | Numeric | Constant | Empty |
|---|---|---|---|---|---|---|---|
| FREYJABM | 160 | 160 (real) | 1,317 | 658 | 627 | 31 | 1 |
| SANTOS Small | 550 | ~50 | 6,358 | 4,346 | 1,323 | 562 | 127 |
| OmniMatch CG | 550 | 11 | 2,641 | 1,422 | 1,146 | 73 | 0 |
| OmniMatch CR | 600 | 12 | 2,657 | 1,527 | 1,119 | 11 | 0 |
| TUS Small | 1,526 | 10 | 14,806 | 13,243 | 1,386 | 177 | 0 |
| **Total** | 3,386 | **~243** | 27,779 | **21,196** | 5,601 | 854 | 128 |

"Original tables" is the number of tables each synthetic lake was derived from, inferred from file names. The column totals match Table III of the FREYJA paper.

**Key finding: real diversity is far smaller than the column count.** The 21,196 text columns come from about 243 original tables, 160 of them from FREYJABM. TUS Small contributes 62 % of the text columns from only 10 tables. Copies must be grouped by original table, otherwise pieces of the same table would land in train and test.

### 2.2 Additional real corpora (2026-10-11)

| Corpus | Source | Tables | Text columns | Typical size |
|---|---|---|---|---|
| **GitTables** (CSV files from GitHub, CC0) | [Zenodo 6515973](https://zenodo.org/records/6515973), 96 topics | 43,167 (6 random topics) | 176,305 | median 33 rows, 11 distinct values |
| **LB-WebTable** (web tables) | [Zenodo 15499092](https://zenodo.org/records/15499092) | 35,173 | 217,852 | median 22 rows, 7 distinct values, 3.4 characters per value |

GitTables solved the diversity problem (thousands of independent real tables). LB-WebTable was discarded: tables too small and values too short.

### 2.3 The GitTables pipeline (2026-10-11 to 2026-10-12)

Full GitTables: 96 topics, 871,195 CSV tables, 26 GB on disk. The source repository of 1,018,649 tables in 28,227 repositories was recovered from the Parquet release (`src/repos_gittables.py`).

Funnel (`src/inventario.py` → `src/repos_gittables.py` → `src/familias.py`):

| Step | Columns | Tables | Families |
|---|---|---|---|
| Text columns in all lakes | 3,824,788 | | |
| Inclusion criterion (D20) | 1,055,964 | 498,161 | 6,813 |
| At most 3 tables per family (D21) | 38,589 | | 6,813 |
| Without columns of identical values | **26,648** | **10,062** | **6,813** |

- **The largest families are single repositories**: the largest has 249,202 tables, 248,822 of them from one repository.
- **Headerless tables:** 52,041 of the 498,161 included (10 %). Detected, and their first row treated as data.
- **Quality checks.** A random sample of 400 columns revealed three problems: headerless tables, families not detected (1,151 nearly identical columns named `msvcrt.dll` spread across families), and content that is not meaningful text (dates, near-numeric strings, hex addresses). The first two were fixed by detecting headers and grouping by source repository. The third was left as is (D22). A second review confirmed the fixes.
- **Clean folder** (`src/lake_limpio.py`): 10,062 tables in a single format, 0.58 GB. Verified: the number of rows, non-null and distinct values of all 26,648 columns matched the inventory exactly.
- **Split** (`src/particion.py`): train 4,643 families (17,941 columns), validation 995 (3,827), test 995 (3,671), and an external test of 180 families (1,209 columns) from the other lakes. No family crosses parts.

**Why it was replaced.** After phases 1 and 2 my supervisors considered GitTables valid but doubted its variety (96 % of the content comes from GitHub: logs, test results, software corpora, far from an enterprise data lake or an open-data portal). LakeBench OpenData is real, varied, of a different origin and close to the kind of data in FREYJABM, so I switched to it (D33) and freed about 22 GB. The GitTables scripts are kept in `src/` as part of the record, but they no longer have input data.

### 2.4 Other candidates considered

| Candidate | Assessment |
|---|---|
| OmniMatch (CG and CR) | Municipal data, synthetic from 23 original tables. Little real variety. Useful as an external test, not for training |
| LakeBench (another project with the same name, [Zenodo 10042019](https://zenodo.org/records/10042019)) | `ckan-subset` (CKAN open-data portals, 1.5 GB) is another source of real open data |

## 3. Known risks and biases

**Treated.**
- *Edited copies of the same data in different sources:* not grouped if no column is identical. To be measured with the embeddings (D24).
- *Different reading between my code and FREYJA* (malformed rows, separator, encoding, headers, repeated names): handled by matching FREYJA's name cleaning and by checking distinct-value counts.

**Not treated.**
- Tables badly split into columns, if separator detection fails.
- Values that mean "empty" ("NA", "null", "-", "no data") are counted as normal values.
- Trivial variants (trailing spaces, capitalisation) prevent detecting near-identical columns.
- Very long text, which the embedding model truncates.

**Biases by design (scope limits).**
- Only text columns (D18) with at least 50 non-null and 20 distinct values (D20, D25). On FREYJABM, every column FREYJA profiles is used (D29).
- *Popular domains are under-represented:* tables sharing an identical column (the list of countries) end up in the same family, and only one table per family is kept. This reduces the pairs with the highest cosine.
