"""Similitud entre palabras de nombres de columna, aprendida de GitTables.

Idea (docs/methodology.md, ideas guardadas): si dos columnas de orígenes distintos comparten
gran parte de sus valores, sus nombres probablemente significan lo mismo. Por
ejemplo, `Country` y `Nation` contienen Spain, France, ... Se aprende a nivel
de palabra, partiendo cada nombre (`CountryRegionCode` -> country, region, code),
para que sirva con nombres que no aparezcan tal cual en GitTables.

Datos: solo columnas de GitTables de las particiones train, validacion y test
(familias sin relación con los lakes externos, D27). Además se descarta toda
columna de GitTables cuyo contenido coincida en gran parte con alguna columna
de FREYJABM, para que el diccionario no aprenda de copias de FREYJABM.

Salida: data/experimento/sinonimos_gittables.csv con columnas palabra_a,
palabra_b, similitud (en los dos órdenes).

Uso:  python src/sinonimos.py
"""
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
COLUMNAS = DATOS / "columnas.csv"
LAKE_LIMPIO = DATOS / "lake"
FREYJABM = DATOS / "raw/freyjabm/datalake"
SALIDA = DATOS / "experimento/sinonimos_gittables.csv"

MAX_COLUMNAS_POR_VALOR = 200   # valores muy comunes ("yes", "1", "na") no indican nada
MIN_VALORES_COMPARTIDOS = 5
MIN_CONTENCION = 0.5           # parte de los valores de la columna menor presentes en la otra
CONTENCION_COPIA = 0.8         # a partir de aquí, una columna se considera copia de FREYJABM
MIN_VECES = 2                  # coincidencias mínimas para relacionar dos palabras
SUAVIZADO = 5


def palabras(nombre):
    """Parte un nombre de columna en palabras: 'CountryRegionCode' -> {country, region, code}."""
    nombre = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(nombre))      # camelCase
    nombre = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", nombre)     # 'IDNumber' -> 'ID Number'
    trozos = re.split(r"[^A-Za-z]+", nombre.lower())
    return {t for t in trozos if len(t) >= 2}


def valores(serie):
    return set(serie.dropna().astype(str).str.strip().str.lower()) - {""}


def leer_columnas_gittables():
    seleccion = pd.read_csv(COLUMNAS)
    seleccion = seleccion[(seleccion["lake"] == "gittables")
                          & seleccion["particion"].isin(["train", "validacion", "test"])]
    columnas = []
    for archivo, grupo in seleccion.groupby("archivo_limpio"):
        tabla = pd.read_csv(LAKE_LIMPIO / archivo, dtype=str, keep_default_na=False,
                            na_values=[""])
        for fila in grupo.itertuples():
            nombre = "" if fila.sin_cabecera else fila.columna  # sin cabecera no hay nombre
            columnas.append({"nombre": nombre, "familia": fila.familia,
                             "valores": valores(tabla[fila.columna_limpia])})
    return columnas


def leer_valores_freyjabm():
    conjuntos = []
    for archivo in sorted(FREYJABM.glob("*.csv")):
        tabla = pd.read_csv(archivo, dtype=str, encoding="latin1", keep_default_na=False,
                            na_values=[""], on_bad_lines="skip")
        for nombre in tabla.columns:
            conjunto = valores(tabla[nombre])
            if len(conjunto) >= MIN_VALORES_COMPARTIDOS:
                conjuntos.append(conjunto)
    return conjuntos


def quitar_copias_de_freyjabm(columnas, conjuntos_freyjabm):
    indice = defaultdict(list)
    for i, conjunto in enumerate(conjuntos_freyjabm):
        for valor in conjunto:
            indice[valor].append(i)
    limpias = []
    for columna in columnas:
        compartidos = Counter(i for v in columna["valores"] for i in indice.get(v, ()))
        es_copia = any(
            n >= MIN_VALORES_COMPARTIDOS
            and n / min(len(columna["valores"]), len(conjuntos_freyjabm[i])) >= CONTENCION_COPIA
            for i, n in compartidos.items())
        if not es_copia:
            limpias.append(columna)
    return limpias


def parejas_con_contenido_comun(columnas):
    """Parejas de columnas de familias distintas que comparten gran parte de sus valores."""
    indice = defaultdict(list)
    for i, columna in enumerate(columnas):
        for valor in columna["valores"]:
            indice[valor].append(i)
    compartidos = Counter()
    for lista in indice.values():
        if 2 <= len(lista) <= MAX_COLUMNAS_POR_VALOR:
            compartidos.update(combinations(lista, 2))
    parejas = []
    for (i, j), n in compartidos.items():
        a, b = columnas[i], columnas[j]
        if a["familia"] == b["familia"] or n < MIN_VALORES_COMPARTIDOS:
            continue
        if n / min(len(a["valores"]), len(b["valores"])) >= MIN_CONTENCION:
            parejas.append((i, j))
    return parejas


def main():
    columnas = leer_columnas_gittables()
    print(f"Columnas de GitTables: {len(columnas)}")
    columnas = quitar_copias_de_freyjabm(columnas, leer_valores_freyjabm())
    print(f"Tras quitar copias de FREYJABM: {len(columnas)}")

    palabras_de = [palabras(c["nombre"]) for c in columnas]
    columnas_con_palabra = Counter(p for ps in palabras_de for p in ps)

    parejas = parejas_con_contenido_comun(columnas)
    print(f"Parejas de columnas con contenido común (familias distintas): {len(parejas)}")

    juntas = Counter()
    for i, j in parejas:
        for a in palabras_de[i]:
            for b in palabras_de[j]:
                if a != b:
                    juntas[(a, b)] += 1
                    juntas[(b, a)] += 1

    filas = []
    for (a, b), n in juntas.items():
        if n < MIN_VECES:
            continue  # una coincidencia suelta no basta para decir que son sinónimos
        # Normalizado por la frecuencia de cada palabra; SUAVIZADO evita que dos
        # palabras rarísimas salgan con similitud 1 por una sola coincidencia
        similitud = n / np.sqrt((columnas_con_palabra[a] + SUAVIZADO)
                                * (columnas_con_palabra[b] + SUAVIZADO))
        filas.append({"palabra_a": a, "palabra_b": b, "veces": n,
                      "similitud": min(1.0, similitud)})
    tabla = pd.DataFrame(filas).sort_values("similitud", ascending=False)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA, index=False)

    print(f"Parejas de palabras con similitud: {len(tabla) // 2}")
    print(f"Palabras distintas en los nombres: {len(columnas_con_palabra)}")
    print("Ejemplos (con al menos 3 coincidencias):")
    print(tabla[(tabla["veces"] >= 3) & (tabla["palabra_a"] < tabla["palabra_b"])]
          .head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
