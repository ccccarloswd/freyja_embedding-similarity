"""Selección de tablas de LakeBench OpenData para la siguiente ronda (D33).

Parte del inventario de OpenData (data/inventario_lb_opendata.csv) y:
  1. Agrupa las tablas en familias: dos tablas son de la misma familia si tienen
     exactamente la misma cabecera o si comparten alguna columna de texto con
     exactamente los mismos valores (solo columnas con al menos 50 valores y
     20 distintos, para no unir tablas por columnas triviales como "Sí/No").
  2. Baraja las familias al azar (semilla fija) y elige una tabla de cada una
     hasta reunir OBJETIVO_COLUMNAS columnas de texto.
  3. Copia las tablas elegidas, sin modificarlas, a data/raw/lb_opendata/seleccion/,
     que es la carpeta que se le pasa a la API de FREYJA.

Salida: data/seleccion_opendata.csv (tabla, país, familia y columnas de texto).

Uso:  python src/seleccion_opendata.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path(__file__).resolve().parent.parent / "data"
INVENTARIO = DATOS / "inventario_lb_opendata.csv"
ORIGEN = DATOS / "raw/lb_opendata/opendata_sampled_1k/datalake"
DESTINO = DATOS / "raw/lb_opendata/seleccion"
SALIDA = DATOS / "seleccion_opendata.csv"

SEMILLA = 0
OBJETIVO_COLUMNAS = 5000
MIN_NO_NULOS, MIN_DISTINTOS = 50, 20  # columnas que cuentan para unir familias por contenido


def asignar_familias(inventario):
    padre = {}

    def raiz(x):
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for tabla, cabecera in inventario.drop_duplicates("tabla")[["tabla", "cab"]].itertuples(index=False):
        padre[raiz(("tabla", tabla))] = raiz(("cabecera", cabecera))
    relevantes = inventario[(inventario["tipo"] == "texto")
                            & (inventario["no_nulos"] >= MIN_NO_NULOS)
                            & (inventario["distintos"] >= MIN_DISTINTOS)]
    for tabla, huella in relevantes[["tabla", "huella"]].itertuples(index=False):
        padre[raiz(("tabla", tabla))] = raiz(("huella", huella))

    tablas = inventario["tabla"].drop_duplicates()
    return dict(zip(tablas, pd.factorize(pd.Series([str(raiz(("tabla", t))) for t in tablas]))[0]))


def main():
    inventario = pd.read_csv(INVENTARIO, keep_default_na=False)
    familia = asignar_familias(inventario)

    tablas = (inventario.groupby("tabla")
              .agg(pais=("pais", "first"), texto=("tipo", lambda t: (t == "texto").sum()))
              .reset_index())
    tablas["pais"] = tablas["pais"].str.rstrip("_")
    tablas["familia"] = tablas["tabla"].map(familia)
    tablas = tablas[tablas["texto"] > 0]
    print(f"Tablas con columnas de texto: {len(tablas)} en {tablas['familia'].nunique()} familias")

    azar = np.random.default_rng(SEMILLA)
    familias = np.array(sorted(tablas["familia"].unique()))
    azar.shuffle(familias)
    elegidas, columnas = [], 0
    for f in familias:
        candidatas = tablas[tablas["familia"] == f].sort_values("tabla")
        tabla = candidatas.iloc[azar.integers(len(candidatas))]
        elegidas.append(tabla)
        columnas += tabla["texto"]
        if columnas >= OBJETIVO_COLUMNAS:
            break
    seleccion = pd.DataFrame(elegidas).reset_index(drop=True)

    DESTINO.mkdir(parents=True, exist_ok=True)
    for archivo in seleccion["tabla"]:
        shutil.copy2(ORIGEN / archivo, DESTINO / archivo)
    seleccion.rename(columns={"texto": "columnas_texto"}).to_csv(SALIDA, index=False)

    print(f"Elegidas: {len(seleccion)} tablas, {seleccion['texto'].sum()} columnas de texto")
    print("Por país (tablas):", seleccion["pais"].value_counts().to_dict())
    print(f"Copiadas a {DESTINO}")


if __name__ == "__main__":
    sys.exit(main())
