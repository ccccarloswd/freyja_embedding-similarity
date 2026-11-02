"""Reparto de las columnas en train, validación, test y prueba externa.

Añade la columna `particion` a data/columnas.csv (decisiones D7 y D27):

  - externo:    todas las familias con alguna tabla que no sea de GitTables
                (FREYJABM, SANTOS, TUS, OmniMatch). Nunca se usan para entrenar.
  - train, validacion, test: las familias de GitTables, repartidas al azar
                con semilla fija en las proporciones indicadas.

La unidad de reparto es la familia: todas las columnas de una familia van a la
misma parte. Los pares se forman después, solo dentro de cada parte.

Uso:  python src/particion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from familias import SALIDA as COLUMNAS  # noqa: E402

SEMILLA = 0
PROPORCIONES = {"train": 0.70, "validacion": 0.15, "test": 0.15}


def main():
    columnas = pd.read_csv(COLUMNAS).drop(columns=["particion"], errors="ignore")

    externas = set(columnas.loc[columnas["lake"] != "gittables", "familia"])
    internas = np.array(sorted(set(columnas["familia"]) - externas))
    np.random.default_rng(SEMILLA).shuffle(internas)

    cortes = (np.cumsum(list(PROPORCIONES.values())) * len(internas)).round().astype(int)
    partes = np.split(internas, cortes[:-1])
    particion = {familia: "externo" for familia in externas}
    for nombre, familias in zip(PROPORCIONES, partes):
        particion.update({familia: nombre for familia in familias})

    columnas["particion"] = columnas["familia"].map(particion)
    columnas.to_csv(COLUMNAS, index=False)

    resumen = columnas.groupby("particion").agg(familias=("familia", "nunique"),
                                                tablas=("archivo_limpio", "nunique"),
                                                columnas=("columna", "size"))
    print(resumen.loc[[*PROPORCIONES, "externo"]])
    print("\nLakes en cada parte:")
    print(pd.crosstab(columnas["particion"], columnas["lake"]))
    repetidas = columnas.groupby("familia")["particion"].nunique().gt(1).sum()
    print(f"\nFamilias repartidas en más de una parte: {repetidas}")


if __name__ == "__main__":
    sys.exit(main())
