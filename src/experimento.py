"""Fase 1: ¿se puede predecir el cosine entre embeddings a partir de los perfiles?

Uso: python src/experimento.py <variante> <modelo de embedding>
     por ejemplo: python src/experimento.py v1 mpnet

Variantes:
  v1   las 34 distancias de FREYJA + name_dist
  v2a  v1 + parecido de los nombres palabra a palabra (iguales o abreviaturas)
  v2b  v2a + parecido de los nombres usando también los sinónimos aprendidos de GitTables

Modelos de embedding: warpgate (por defecto) o mpnet.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

VARIANTE = sys.argv[1] if len(sys.argv) > 1 else "v1"
MODELO_EMBEDDING = sys.argv[2] if len(sys.argv) > 2 else "warpgate"

DATOS = Path(__file__).resolve().parent.parent / "data"
PERFILES = DATOS / "perfiles/freyjabm/profiles_freyjabm_normalized.pkl"
EMBEDDINGS = DATOS / f"embeddings/freyjabm_{MODELO_EMBEDDING}/original.npy"
COLUMNAS_DE_LOS_EMBEDDINGS = DATOS / f"embeddings/freyjabm_{MODELO_EMBEDDING}/columnas.csv"
DISTANCIAS = DATOS / "experimento/distancias_freyjabm.csv"
SINONIMOS = DATOS / "experimento/sinonimos_gittables.csv"
PREDICCIONES = DATOS / f"experimento/predicciones_freyjabm_{MODELO_EMBEDDING}_{VARIANTE}.csv"
IMPORTANCIAS = DATOS / f"experimento/importancias_freyjabm_{MODELO_EMBEDDING}_{VARIANTE}.csv"

SEMILLA = 0  # para que el reparto "al azar" sea siempre el mismo
GRUPOS = 5   # número de grupos de la validación cruzada

MIN_CANDIDATAS = 5  # columnas consulta con menos candidatas no cuentan en el Spearman

# Columnas de la tabla de pares que NO son entrada del modelo
NO_SON_VARIABLES = ["consulta", "candidata", "coseno", "grupo_consulta", "grupo_candidata"]

# Se usa la función de distancias del propio código de FREYJA
# Repo de FREYJA: variable de entorno FREYJA_DIR, o por defecto una carpeta
# FREYJA junto a la de este proyecto
FREYJA = Path(os.environ.get("FREYJA_DIR", Path(__file__).resolve().parents[2] / "FREYJA"))
sys.path.insert(0, str(FREYJA))
from app.core.distances.compute_distances import ComputeDistances, ComputeDistancesConfig  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinonimos import palabras  # noqa: E402  (misma forma de partir nombres que el diccionario)


def cargar_datos():
    """Devuelve los perfiles y, en el mismo orden, el embedding de cada columna."""
    perfiles = pd.read_pickle(PERFILES)
    todos_los_embeddings = np.load(EMBEDDINGS)
    columnas = pd.read_csv(COLUMNAS_DE_LOS_EMBEDDINGS)

    # La fila i de 'columnas' dice de qué columna es la fila i de los embeddings.
    # Se guarda, para cada (tabla, columna), en qué fila está su embedding.
    fila_del_embedding = {}
    for i, columna in columnas.iterrows():
        tabla = columna["tabla"].removesuffix(".csv")
        fila_del_embedding[(tabla, columna["columna"])] = i

    # Se recorren los perfiles y se toma el embedding de cada uno, en el mismo orden
    embeddings = []
    for _, perfil in perfiles.iterrows():
        i = fila_del_embedding[(perfil["dataset_name"], perfil["attribute_name"])]
        embeddings.append(todos_los_embeddings[i])
    embeddings = np.array(embeddings)

    return perfiles, embeddings


def calcular_distancias(perfiles):
    """Distancias de FREYJA para cada par de columnas de tablas distintas, en los dos órdenes."""
    calculadora = ComputeDistances(ComputeDistancesConfig(
        profiles_file_path=PERFILES,
        ground_truth_path=None,
        output_distances_path=DISTANCIAS,
    ))

    trozos = []
    for consulta in range(len(perfiles)):
        # Distancias de la columna 'consulta' a todas las columnas (ella incluida)
        distancias = calculadora._compute_distances_vectorized(perfiles.iloc[consulta], perfiles)
        distancias.insert(0, "consulta", consulta)
        distancias.insert(1, "candidata", range(len(perfiles)))

        # Solo cuentan las candidatas de otras tablas
        tabla_de_la_consulta = perfiles["dataset_name"].iloc[consulta]
        de_otra_tabla = (perfiles["dataset_name"] != tabla_de_la_consulta).to_numpy()
        trozos.append(distancias[de_otra_tabla])

    return pd.concat(trozos, ignore_index=True)


def obtener_distancias(perfiles):
    """Lee las distancias si ya están calculadas; si no, las calcula y las guarda."""
    if DISTANCIAS.exists():
        return pd.read_csv(DISTANCIAS)

    distancias = calcular_distancias(perfiles)
    DISTANCIAS.parent.mkdir(parents=True, exist_ok=True)
    distancias.to_csv(DISTANCIAS, index=False)
    return distancias


def coseno(a, b):
    """Cosine similarity entre dos vectores."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calcular_cosenos(pares, embeddings):
    """Cosine entre los embeddings de las dos columnas de cada par."""
    cosenos = []
    for consulta, candidata in zip(pares["consulta"], pares["candidata"]):
        cosenos.append(coseno(embeddings[consulta], embeddings[candidata]))
    return cosenos


def es_abreviatura(corta, larga):
    """'ctry' es abreviatura de 'country': misma primera letra y letras en orden."""
    if len(corta) >= len(larga) or corta[0] != larga[0]:
        return False
    resto = iter(larga)
    return all(letra in resto for letra in corta)


def parecido_entre_palabras(a, b, sinonimos):
    if a == b or es_abreviatura(a, b) or es_abreviatura(b, a):
        return 1.0
    return sinonimos.get((a, b), 0.0)


def parecido_de_nombres(palabras_a, palabras_b, sinonimos):
    """Para cada palabra de un nombre, su mejor pareja en el otro; media en los dos sentidos."""
    if not palabras_a or not palabras_b:
        return 0.0
    ida = np.mean([max(parecido_entre_palabras(a, b, sinonimos) for b in palabras_b)
                   for a in palabras_a])
    vuelta = np.mean([max(parecido_entre_palabras(b, a, sinonimos) for a in palabras_a)
                      for b in palabras_b])
    return (ida + vuelta) / 2


def calcular_parecido_de_nombres(pares, perfiles, sinonimos):
    palabras_de = [palabras(nombre) for nombre in perfiles["attribute_name"]]
    return [parecido_de_nombres(palabras_de[i], palabras_de[j], sinonimos)
            for i, j in zip(pares["consulta"], pares["candidata"])]


def leer_sinonimos():
    tabla = pd.read_csv(SINONIMOS)
    return dict(zip(zip(tabla["palabra_a"], tabla["palabra_b"]), tabla["similitud"]))


def repartir_tablas(perfiles):
    """Asigna cada tabla a uno de los GRUPOS, al azar pero siempre igual (semilla fija)."""
    tablas = sorted(perfiles["dataset_name"].unique())
    generador = np.random.default_rng(SEMILLA)
    generador.shuffle(tablas)

    # Tras barajar, se reparte como cartas: 0, 1, 2, 3, 4, 0, 1, 2, ...
    grupo_de_la_tabla = {}
    for posicion, tabla in enumerate(tablas):
        grupo_de_la_tabla[tabla] = posicion % GRUPOS
    return grupo_de_la_tabla


def crear_modelo():
    """Gradient Boosting con los mismos ajustes que el modelo de FREYJA."""
    return GradientBoostingRegressor(
        n_estimators=100,      # número de árboles
        learning_rate=0.05,    # cuánto corrige cada árbol a los anteriores
        max_depth=3,           # profundidad máxima de cada árbol
        subsample=0.8,         # cada árbol aprende con el 80 % de los pares, al azar
        min_samples_leaf=10,   # cada hoja del árbol debe tener al menos 10 pares
        random_state=SEMILLA,
    )


def separar_ronda(distancias, grupo):
    """Qué pares son de entrenamiento y cuáles de examen cuando 'grupo' hace de examen."""
    consulta_en_grupo = distancias["grupo_consulta"] == grupo
    candidata_en_grupo = distancias["grupo_candidata"] == grupo
    examen = consulta_en_grupo & candidata_en_grupo           # las dos columnas en el grupo
    entrenamiento = ~consulta_en_grupo & ~candidata_en_grupo  # ninguna en el grupo
    return entrenamiento, examen


def entrenar_y_predecir(distancias):
    """En cada ronda, entrena un modelo nuevo, predice los pares de su examen
    y mide la importancia de cada variable."""
    variables = [c for c in distancias.columns if c not in NO_SON_VARIABLES]
    predicciones = pd.Series(np.nan, index=distancias.index)
    importancias = []

    for grupo in range(GRUPOS):
        entrenamiento, examen = separar_ronda(distancias, grupo)
        modelo = crear_modelo()
        modelo.fit(distancias.loc[entrenamiento, variables],
                   distancias.loc[entrenamiento, "coseno"])
        predicciones[examen] = modelo.predict(distancias.loc[examen, variables])

        # Importancia 1: cuánto ha usado el modelo cada variable al construir sus árboles
        importancia_arboles = modelo.feature_importances_

        # Importancia 2: cuánto empeora el R² del examen si se desordena esa variable
        permutacion = permutation_importance(
            modelo, distancias.loc[examen, variables], distancias.loc[examen, "coseno"],
            scoring="r2", n_repeats=5, random_state=SEMILLA)

        importancias.append(pd.DataFrame({
            "ronda": grupo,
            "variable": variables,
            "importancia_arboles": importancia_arboles,
            "importancia_permutacion": permutacion.importances_mean,
        }))
        print(f"Ronda {grupo}: entrenado con {entrenamiento.sum()} pares, "
              f"examinado con {examen.sum()} pares", flush=True)

    return predicciones, pd.concat(importancias, ignore_index=True)


def obtener_predicciones(distancias):
    """Lee predicciones e importancias si ya están calculadas; si no, entrena y las guarda."""
    if PREDICCIONES.exists() and IMPORTANCIAS.exists():
        return pd.read_csv(PREDICCIONES)["prediccion"], pd.read_csv(IMPORTANCIAS)

    predicciones, importancias = entrenar_y_predecir(distancias)
    PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
    predicciones.rename("prediccion").to_csv(PREDICCIONES, index=False)
    importancias.to_csv(IMPORTANCIAS, index=False)
    return predicciones, importancias


def spearman_por_consulta(examen):
    """Para cada columna consulta, ¿ordena el modelo sus candidatas como el cosine real?"""
    valores = []
    for consulta, sus_pares in examen.groupby("consulta"):
        if len(sus_pares) < MIN_CANDIDATAS:
            continue  # con muy pocas candidatas, el orden no dice nada
        rho = spearmanr(sus_pares["coseno"], sus_pares["prediccion"]).correlation
        valores.append(rho)
    return np.mean(valores)


def evaluar_ronda(examen, media_entrenamiento):
    """Las notas del modelo en el examen de una ronda, y las de predecir siempre la media."""
    real = examen["coseno"]
    prediccion = examen["prediccion"]

    error_modelo = real - prediccion
    error_media = real - media_entrenamiento

    return {
        "mae_modelo": error_modelo.abs().mean(),
        "mae_media": error_media.abs().mean(),
        "r2": 1 - (error_modelo ** 2).sum() / (error_media ** 2).sum(),
        "spearman_por_consulta": spearman_por_consulta(examen),
    }


def main():
    # 1. Cargar los datos: perfiles y embeddings de cada columna
    perfiles, embeddings = cargar_datos()
    print("Perfiles:", perfiles.shape)
    print("Embeddings:", embeddings.shape)

    # 2. Formar los pares de columnas y calcular sus distancias
    distancias = obtener_distancias(perfiles)
    print("Pares:", distancias.shape)

    # 3. Calcular el cosine de cada par (lo que se quiere predecir)
    distancias["coseno"] = calcular_cosenos(distancias, embeddings)

    # 3b. Variables nuevas sobre el nombre, según la variante
    if VARIANTE in ("v2a", "v2b"):
        distancias["nombre_palabras"] = calcular_parecido_de_nombres(distancias, perfiles, {})
    if VARIANTE == "v2b":
        distancias["nombre_sinonimos"] = calcular_parecido_de_nombres(
            distancias, perfiles, leer_sinonimos())
    print(f"Variante {VARIANTE}, embeddings de {MODELO_EMBEDDING}")
    print(distancias["coseno"].describe().round(3).to_string())

    # 4. Repartir las tablas en 5 grupos para la validación cruzada
    grupo_de_la_tabla = repartir_tablas(perfiles)

    # El grupo de cada columna es el de su tabla; el de cada par, el de sus dos columnas
    grupo_de_la_columna = [grupo_de_la_tabla[tabla] for tabla in perfiles["dataset_name"]]
    distancias["grupo_consulta"] = [grupo_de_la_columna[i] for i in distancias["consulta"]]
    distancias["grupo_candidata"] = [grupo_de_la_columna[i] for i in distancias["candidata"]]

    for grupo in range(GRUPOS):
        tablas = [t for t, g in grupo_de_la_tabla.items() if g == grupo]
        columnas = grupo_de_la_columna.count(grupo)
        pares = ((distancias["grupo_consulta"] == grupo)
                 & (distancias["grupo_candidata"] == grupo)).sum()
        print(f"Grupo {grupo}: {len(tablas)} tablas, {columnas} columnas, {pares} pares")
    mezclados = (distancias["grupo_consulta"] != distancias["grupo_candidata"]).sum()
    print(f"Pares entre grupos distintos: {mezclados}")

    # 5. Para cada grupo: entrenar con los otros 4 y predecir el examen de ese grupo
    distancias["prediccion"], importancias = obtener_predicciones(distancias)

    # 6. Evaluar cada ronda y mostrar los resultados
    resultados = []
    for grupo in range(GRUPOS):
        entrenamiento, examen = separar_ronda(distancias, grupo)
        media_entrenamiento = distancias.loc[entrenamiento, "coseno"].mean()
        notas = evaluar_ronda(distancias[examen], media_entrenamiento)
        notas["ronda"] = grupo
        resultados.append(notas)
    resultados = pd.DataFrame(resultados).set_index("ronda")

    print("\nResultados por ronda:")
    print(resultados.round(3))
    print("\nMedia de las 5 rondas (desviación típica):")
    for nota in resultados.columns:
        print(f"  {nota:22s} {resultados[nota].mean():.3f} ({resultados[nota].std():.3f})")

    # 7. Qué ha aprendido el modelo: importancia de cada variable (media de las 5 rondas)
    media_importancias = (importancias
                          .groupby("variable")[["importancia_arboles", "importancia_permutacion"]]
                          .mean()
                          .sort_values("importancia_permutacion", ascending=False))
    print("\nImportancia de las variables (media de las 5 rondas):")
    print(media_importancias.round(3).to_string())


if __name__ == "__main__":
    main()
