"""Inventario de columnas de los data lakes.

Recorre todos los CSV de cada data lake y escribe una fila por columna en
data/inventario.csv, con su tamaño y una clasificación orientativa
(vacia, constante, numerica, texto) y si cumple el criterio de inclusión.

La clasificación de tipo es orientativa: la definitiva será la de la API de
FREYJA (que usa DuckDB) cuando se calculen los perfiles.

Uso:  python src/inventario.py
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "inventario.csv"

# Nombre del lake -> carpetas con sus CSV
CRUDO = RAIZ / "data/raw"
LAKES = {
    "freyjabm": [CRUDO / "freyjabm/datalake"],
    "santos_small": [CRUDO / "santos_small/santos_benchmark/datalake"],
    "omnimatch_cg": [CRUDO / "omnimatch/datasets/city_government/train_tables",
                     CRUDO / "omnimatch/datasets/city_government/test_tables"],
    "omnimatch_cr": [CRUDO / "omnimatch/datasets/culture_recreation/train_tables",
                     CRUDO / "omnimatch/datasets/culture_recreation/test_tables"],
    "tus_small": [CRUDO / "tus_small/tus/datalake"],
    # GitTables: una carpeta por tema descargado
    "gittables": sorted(c for c in (CRUDO / "gittables").glob("*") if c.is_dir()),
}

# Criterio de inclusión de columnas (decisión D20 de docs/methodology.md)
MIN_NO_NULOS = 50
MIN_DISTINTOS = 20

CODIFICACIONES = ("utf-8-sig", "latin-1")  # utf-8-sig también quita la marca BOM inicial


def leer_csv(ruta):
    """Lee un CSV con todo como texto. Devuelve (tabla, codificacion, separador)."""
    for codificacion in CODIFICACIONES:
        try:
            with open(ruta, encoding=codificacion, newline="") as f:
                muestra = f.read(20000)
            try:
                separador = csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
            except csv.Error:
                separador = ","
            opciones = dict(dtype=str, encoding=codificacion, keep_default_na=False,
                            na_values=[""], on_bad_lines="skip", low_memory=False)
            tabla = pd.read_csv(ruta, sep=separador, **opciones)
            if tabla.shape[1] == 1 and separador != ",":
                # La detección automática falló: se vuelve a la coma
                separador = ","
                tabla = pd.read_csv(ruta, sep=separador, **opciones)
            return tabla, codificacion, separador
        except UnicodeDecodeError:
            continue
    raise ValueError("no se pudo decodificar")


def clasificar(valores):
    """Clasifica una columna a partir de sus valores no nulos."""
    if len(valores) == 0:
        return "vacia"
    if valores.nunique() == 1:
        return "constante"
    if pd.to_numeric(valores, errors="coerce").notna().all():
        return "numerica"
    return "texto"


def inventariar_tabla(args):
    lake, ruta = args
    try:
        tabla, codificacion, separador = leer_csv(ruta)
    except Exception as error:  # se registra y se sigue con las demás tablas
        return [], (lake, f"{ruta.parent.name}/{ruta.name}", repr(error))
    filas = []
    for posicion, nombre in enumerate(tabla.columns):
        valores = tabla.iloc[:, posicion].dropna()
        filas.append({
            "lake": lake,
            "tabla": f"{ruta.parent.name}/{ruta.name}",
            "columna": nombre,
            "posicion": posicion,
            "filas": len(tabla),
            "no_nulos": len(valores),
            "distintos": valores.nunique(),
            "longitud_media": round(valores.str.len().mean(), 2) if len(valores) else 0,
            "tipo": clasificar(valores),
            "codificacion": codificacion,
            "separador": separador,
        })
    return filas, None


def main():
    tareas = [(lake, ruta) for lake, carpetas in LAKES.items()
              for carpeta in carpetas for ruta in sorted(carpeta.glob("*.csv"))]
    filas, errores = [], []
    with ProcessPoolExecutor(max_workers=8) as ejecutor:
        for filas_tabla, error in ejecutor.map(inventariar_tabla, tareas, chunksize=64):
            filas.extend(filas_tabla)
            if error:
                errores.append(error)

    inventario = pd.DataFrame(filas)
    inventario["incluida"] = ((inventario["tipo"] == "texto")
                              & (inventario["no_nulos"] >= MIN_NO_NULOS)
                              & (inventario["distintos"] >= MIN_DISTINTOS))
    inventario.to_csv(SALIDA, index=False)

    print(f"Tablas leídas: {inventario.groupby('lake')['tabla'].nunique().to_dict()}")
    print(f"Tablas con error: {len(errores)}")
    for error in errores:
        print("  ", error)
    print("\nColumnas por lake y tipo:")
    print(pd.crosstab(inventario["lake"], inventario["tipo"], margins=True))
    incluidas = inventario[inventario["incluida"]]
    print(f"\nColumnas incluidas (texto, >={MIN_NO_NULOS} no nulos, >={MIN_DISTINTOS} distintos):")
    print(incluidas.groupby("lake").agg(columnas=("columna", "size"), tablas=("tabla", "nunique")))
    print("Total:", len(incluidas), "columnas de", incluidas.groupby(["lake", "tabla"]).ngroups, "tablas")
    texto = inventario[inventario["tipo"] == "texto"]
    print("\nColumnas de texto: filas, valores distintos y longitud media (cuartiles):")
    print(texto.groupby("lake")[["filas", "distintos", "longitud_media"]]
          .describe(percentiles=[0.25, 0.5, 0.75]).round(1).T.to_string())
    print("\nSeparadores y codificaciones (por tabla):")
    print(inventario.drop_duplicates(["lake", "tabla"])
          .groupby(["lake", "separador", "codificacion"]).size())
    print(f"\nInventario escrito en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
