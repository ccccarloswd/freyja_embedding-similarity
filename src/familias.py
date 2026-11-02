"""Selección final de columnas: familias de tablas, límite por familia y duplicados.

Parte de data/inventario.csv (y de data/gittables_repos.csv) y escribe
data/columnas.csv, con una fila por columna seleccionada y la familia a la que
pertenece su tabla.

Pasos (decisiones D5, D20 y D21 de docs/methodology.md):
  1. Tablas sin cabecera: se detectan y su primera fila se trata como datos.
  2. Familias: dos tablas son de la misma familia si
       a) proceden de la misma tabla original (lakes sintéticos, por el nombre
          de archivo), o
       b) proceden del mismo repositorio de GitHub (GitTables), o
       c) tienen exactamente la misma cabecera (solo tablas con cabecera), o
       d) comparten alguna columna con exactamente los mismos valores.
  3. Límite: como máximo MAX_TABLAS_POR_FAMILIA tablas por familia, elegidas
     al azar con semilla fija.
  4. Duplicados: de las columnas con exactamente los mismos valores solo se
     conserva una.

La familia es la unidad que después se reparte entre train y test.

Uso:  python src/familias.py
"""
import hashlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inventario import LAKES, RAIZ, leer_csv  # noqa: E402

ENTRADA = RAIZ / "data" / "inventario.csv"
REPOSITORIOS = RAIZ / "data" / "gittables_repos.csv"
SALIDA = RAIZ / "data" / "columnas.csv"

MAX_TABLAS_POR_FAMILIA = 3
SEMILLA = 0


# ---------------------------------------------------------------- cabeceras

def forma(texto):
    """Forma de un valor: '2015-11-19 21:52' -> '9-9-9 9:9', 'abc12' -> 'a9'."""
    texto = re.sub(r"\d+", "9", str(texto).strip())
    texto = re.sub(r"[^\W\d_]+", "a", texto)
    return re.sub(r"\s+", " ", texto)


def parece_sin_cabecera(tabla):
    """Indica si la primera fila del CSV parece una fila de datos y no una cabecera.

    Se considera que no hay cabecera si, en alguna columna, el supuesto nombre
      - aparece también como valor de esa columna, o
      - tiene la misma forma que la mayoría de sus valores, siendo esa forma
        claramente de dato (con cifras y al menos tres símbolos, como una
        fecha '9-9-9' o un decimal '9.9'; un entero o 'a9' no bastan).
    """
    for posicion, nombre in enumerate(tabla.columns):
        valores = tabla.iloc[:, posicion].dropna()
        if len(valores) == 0:
            continue
        nombre = str(nombre)
        if nombre in set(valores):
            return True
        forma_nombre = forma(nombre)
        if "9" in forma_nombre and len(forma_nombre) >= 3:
            if (valores.map(forma) == forma_nombre).mean() >= 0.5:
                return True
    return False


def leer_tabla(lake, tabla):
    """Lee una tabla tratando bien la cabecera. Devuelve (datos, sin_cabecera)."""
    ruta = ruta_de(lake, tabla)
    datos, codificacion, separador = leer_csv(ruta)
    if not parece_sin_cabecera(datos):
        return datos, False
    completa = pd.read_csv(ruta, sep=separador, header=None, dtype=str, encoding=codificacion,
                           keep_default_na=False, na_values=[""], on_bad_lines="skip",
                           low_memory=False)
    if len(completa) < len(datos) or completa.shape[1] != datos.shape[1]:
        # Releer sin cabecera rompe la tabla (p. ej. comillas sin cerrar en la
        # primera fila): se conserva la lectura con cabecera
        return datos, False
    return completa, True


def ruta_de(lake, tabla):
    carpeta, archivo = tabla.split("/")
    return next(c for c in LAKES[lake] if c.name == carpeta) / archivo


# ----------------------------------------------------------------- familias

def tabla_original(lake, tabla):
    """Nombre de la tabla original de la que procede un trozo (lakes sintéticos)."""
    archivo = tabla.split("/")[1].removesuffix(".csv")
    if lake == "tus_small":            # t_1934eacab8c57857____c10_0____0
        return archivo.split("____")[0]
    if lake.startswith("omnimatch"):   # evictions_6_noisy_2 -> evictions
        return re.sub(r"(_\d+)?_noisy_\d+$", "", archivo)
    if lake == "santos_small":         # business_rates_8 -> business_rates
        return re.sub(r"_[0-9a-z]{1,2}$", "", archivo)
    return None


def huellas_de_tabla(args):
    """Huella del contenido (valores no nulos, sin importar el orden) de cada columna."""
    lake, tabla, posiciones = args
    datos, sin_cabecera = leer_tabla(lake, tabla)
    huellas = []
    for posicion in posiciones:
        if posicion >= datos.shape[1]:
            continue
        valores = sorted(datos.iloc[:, posicion].dropna())
        huella = hashlib.md5("\x1f".join(valores).encode("utf-8")).hexdigest()
        huellas.append((lake, tabla, posicion, huella, sin_cabecera))
    return huellas


def asignar_familias(inventario, huellas, repositorios):
    """Devuelve una tabla (lake, tabla, familia) uniendo los cuatro criterios."""
    padre = {}

    def raiz(x):
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a, b):
        padre[raiz(a)] = raiz(b)

    tablas = huellas[["lake", "tabla", "sin_cabecera"]].drop_duplicates()
    cabeceras = (inventario.merge(tablas[["lake", "tabla"]], on=["lake", "tabla"])
                 .groupby(["lake", "tabla"])["columna"]
                 .apply(lambda c: "|".join(sorted(str(x).strip().lower() for x in c))))

    for fila in tablas.itertuples():
        nodo = ("tabla", fila.lake, fila.tabla)
        raiz(nodo)
        original = tabla_original(fila.lake, fila.tabla)
        if original is not None:                                   # criterio a
            unir(nodo, ("original", fila.lake, original))
        repositorio = repositorios.get(fila.tabla) if fila.lake == "gittables" else None
        if repositorio:                                            # criterio b
            unir(nodo, ("repositorio", repositorio))
        if not fila.sin_cabecera:                                  # criterio c
            unir(nodo, ("cabecera", cabeceras[(fila.lake, fila.tabla)]))
    for fila in huellas.itertuples():                              # criterio d
        unir(("tabla", fila.lake, fila.tabla), ("huella", fila.huella))

    raices = [str(raiz(("tabla", f.lake, f.tabla))) for f in tablas.itertuples()]
    tablas = tablas.assign(familia=pd.factorize(pd.Series(raices))[0])
    return tablas[["lake", "tabla", "familia"]]


def limitar_tablas(columnas):
    """Conserva como máximo MAX_TABLAS_POR_FAMILIA tablas por familia."""
    azar = np.random.default_rng(SEMILLA)
    tablas = columnas[["familia", "lake", "tabla"]].drop_duplicates().sort_values(
        ["familia", "lake", "tabla"])
    elegidas = []
    for _, grupo in tablas.groupby("familia", sort=True):
        if len(grupo) > MAX_TABLAS_POR_FAMILIA:
            grupo = grupo.iloc[np.sort(azar.choice(len(grupo), MAX_TABLAS_POR_FAMILIA,
                                                   replace=False))]
        elegidas.append(grupo)
    return columnas.merge(pd.concat(elegidas)[["lake", "tabla"]], on=["lake", "tabla"])


# --------------------------------------------------------------------- main

def main():
    inventario = pd.read_csv(ENTRADA, low_memory=False)
    columnas = inventario[inventario["incluida"]]

    repositorios = pd.read_csv(REPOSITORIOS, keep_default_na=False)
    repositorios = dict(zip(repositorios["tema"] + "/" + repositorios["archivo"],
                            repositorios["repositorio"]))

    tareas = [(lake, tabla, grupo["posicion"].tolist())
              for (lake, tabla), grupo in columnas.groupby(["lake", "tabla"])]
    with ProcessPoolExecutor(max_workers=12) as ejecutor:
        huellas = [h for tabla in ejecutor.map(huellas_de_tabla, tareas, chunksize=64)
                   for h in tabla]
    huellas = pd.DataFrame(huellas, columns=["lake", "tabla", "posicion", "huella",
                                             "sin_cabecera"])
    tablas = huellas.drop_duplicates(["lake", "tabla"])
    print(f"Tablas sin cabecera: {tablas['sin_cabecera'].sum()} de {len(tablas)}")
    en_git = tablas[tablas["lake"] == "gittables"]
    print(f"Tablas de GitTables sin repositorio conocido: "
          f"{(~en_git['tabla'].map(repositorios).fillna('').astype(bool)).sum()}")

    familias = asignar_familias(inventario, huellas, repositorios)
    columnas = columnas.merge(huellas, on=["lake", "tabla", "posicion"])
    columnas = columnas.merge(familias, on=["lake", "tabla"])
    print(f"Columnas incluidas: {len(columnas)} en {columnas['familia'].nunique()} familias")
    tablas_por_familia = columnas.groupby("familia")["tabla"].nunique().sort_values()
    print(f"Familias de una sola tabla: {(tablas_por_familia == 1).sum()}")
    print("Las cinco familias mayores (tablas, repositorios):")
    for familia in tablas_por_familia.index[-5:][::-1]:
        de_familia = columnas[columnas["familia"] == familia].drop_duplicates("tabla")
        repos = de_familia["tabla"].map(repositorios).dropna()
        print(f"  {len(de_familia)} tablas, {repos.nunique()} repositorios; "
              f"más frecuentes: {repos.value_counts().head(3).to_dict()}")

    columnas = limitar_tablas(columnas)
    print(f"Tras el límite de {MAX_TABLAS_POR_FAMILIA} tablas por familia: {len(columnas)}")

    columnas = columnas.sort_values(["lake", "tabla", "posicion"])
    duplicadas = columnas.duplicated("huella", keep="first")
    print(f"Columnas con valores idénticos a otra (se eliminan): {duplicadas.sum()}")
    columnas = columnas[~duplicadas]

    columnas = columnas[["lake", "tabla", "columna", "posicion", "familia", "sin_cabecera",
                         "filas", "no_nulos", "distintos", "longitud_media"]]
    columnas.to_csv(SALIDA, index=False)

    print("\nSelección final:")
    resumen = columnas.groupby("lake").agg(columnas=("columna", "size"),
                                           tablas=("tabla", "nunique"),
                                           familias=("familia", "nunique"))
    resumen.loc["Total"] = [len(columnas), columnas.groupby(["lake", "tabla"]).ngroups,
                            columnas["familia"].nunique()]
    print(resumen)
    por_familia = columnas.groupby("familia").size()
    print(f"\nColumnas por familia: mediana {por_familia.median():.0f}, "
          f"máximo {por_familia.max()}")
    print(f"Escrito en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
