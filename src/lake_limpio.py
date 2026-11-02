"""Carpeta limpia con las tablas seleccionadas.

Copia las tablas de data/columnas.csv a data/lake/, todas con el mismo formato,
para que los perfiles (API de FREYJA) y los embeddings lean exactamente los
mismos valores:

  - un archivo por tabla, con nombre t00001.csv, t00002.csv, ...
  - UTF-8, separado por comas, textos entre comillas, valores vacíos como nulos
  - siempre con cabecera, y con nombres de columna c0, c1, ... (la posición
    original), únicos y sin caracteres problemáticos
  - las tablas sin cabecera conservan su primera fila como datos

Se copian las tablas completas (todas sus columnas), no solo las columnas
seleccionadas. A data/columnas.csv se le añaden `archivo_limpio` y
`columna_limpia`, y se recalculan `filas`, `no_nulos` y `distintos` sobre lo
escrito, para poder contrastarlos después con los perfiles de FREYJA.

Uso:  python src/lake_limpio.py
"""
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from familias import SALIDA as COLUMNAS, leer_tabla  # noqa: E402
from inventario import RAIZ  # noqa: E402

LAKE = RAIZ / "data" / "lake"


def copiar_tabla(args):
    lake, tabla, archivo, posiciones = args
    datos, sin_cabecera = leer_tabla(lake, tabla)
    datos.columns = [f"c{i}" for i in range(datos.shape[1])]
    # Todos los textos entre comillas: así los saltos de línea dentro de un valor
    # no parten la fila. Los nulos quedan como campo vacío sin comillas.
    datos.to_csv(LAKE / archivo, index=False, encoding="utf-8", lineterminator="\n",
                 quoting=csv.QUOTE_NONNUMERIC)
    filas = []
    for posicion in posiciones:
        valores = datos.iloc[:, posicion].dropna()
        filas.append((lake, tabla, posicion, len(datos), len(valores), valores.nunique(),
                      sin_cabecera))
    return filas


def main():
    columnas = pd.read_csv(COLUMNAS)
    columnas = columnas.drop(columns=["archivo_limpio", "columna_limpia"], errors="ignore")

    tablas = columnas[["lake", "tabla"]].drop_duplicates().sort_values(["lake", "tabla"])
    tablas["archivo_limpio"] = [f"t{i:05d}.csv" for i in range(1, len(tablas) + 1)]
    columnas = columnas.merge(tablas, on=["lake", "tabla"])
    columnas["columna_limpia"] = "c" + columnas["posicion"].astype(str)

    if LAKE.exists():
        shutil.rmtree(LAKE)  # la carpeta se regenera entera
    LAKE.mkdir(parents=True)

    tareas = [(lake, tabla, archivo, grupo["posicion"].tolist())
              for (lake, tabla, archivo), grupo
              in columnas.groupby(["lake", "tabla", "archivo_limpio"])]
    with ProcessPoolExecutor(max_workers=12) as ejecutor:
        medidas = [m for tabla in ejecutor.map(copiar_tabla, tareas, chunksize=16)
                   for m in tabla]
    medidas = pd.DataFrame(medidas, columns=["lake", "tabla", "posicion", "filas",
                                             "no_nulos", "distintos", "sin_cabecera"])

    antes = columnas[["lake", "tabla", "posicion", "no_nulos", "distintos", "sin_cabecera"]]
    columnas = (columnas.drop(columns=["filas", "no_nulos", "distintos", "sin_cabecera"])
                .merge(medidas, on=["lake", "tabla", "posicion"]))
    cambio = antes.merge(medidas, on=["lake", "tabla", "posicion"], suffixes=("_a", "_d"))

    columnas = columnas[["lake", "tabla", "columna", "posicion", "familia", "sin_cabecera",
                         "archivo_limpio", "columna_limpia", "filas", "no_nulos",
                         "distintos", "longitud_media"]]
    columnas.to_csv(COLUMNAS, index=False)

    archivos = list(LAKE.glob("*.csv"))
    print(f"Tablas escritas: {len(archivos)} "
          f"({sum(a.stat().st_size for a in archivos) / 1e9:.2f} GB) en {LAKE}")
    print(f"Columnas seleccionadas: {len(columnas)}")
    print(f"Columnas cuyo número de valores distintos cambia respecto al inventario: "
          f"{(cambio['distintos_a'] != cambio['distintos_d']).sum()}")
    print(f"  de ellas, en tablas sin cabecera: "
          f"{((cambio['distintos_a'] != cambio['distintos_d']) & cambio['sin_cabecera_d']).sum()}")
    print(f"Tablas cuya detección de cabecera difiere de la anterior: "
          f"{cambio[cambio['sin_cabecera_a'] != cambio['sin_cabecera_d']]['tabla'].nunique()}")


if __name__ == "__main__":
    sys.exit(main())
