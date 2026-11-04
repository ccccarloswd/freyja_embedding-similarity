"""Criba de modelos con OpenData (D34, D35).

Un único reparto por tablas: 60 % entrenamiento, 20 % validación, 20 % test.
  - Entrenamiento y validación: cada columna se empareja con hasta CANDIDATAS
    columnas al azar de otras tablas de su misma parte, en los dos órdenes (D10).
  - Test: todos los pares de columnas de tablas distintas de la parte de test
    (distribución natural, para medir el ranking por columna consulta).
Las distancias son las de FREYJA (34 métricas + name_dist). Los pares no
dependen del embedding y se guardan para reutilizarlos.

Cada modelo se prueba con 3 configuraciones (A por defecto, B contenida,
C amplia). La configuración se elige con validación (Spearman por consulta,
D15); los modelos de boosting y RealMLP usan además la validación para decidir
cuándo parar. Después se evalúa en test. El test no interviene en ninguna decisión.

Uso (entorno .venv-ml):  python src/criba.py <warpgate|mpnet> [A|todas] [modelos a saltar]
  A:     solo la configuración por defecto de cada modelo (para medir tiempos)
  todas: las tres configuraciones
Resultados en data/criba/: se guardan tras cada modelo y configuración; al
volver a ejecutar se saltan los ya hechos.
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from pytabkit import RealMLP_TD_Regressor
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

EMBEDDING = sys.argv[1] if len(sys.argv) > 1 else "warpgate"
ETAPA = sys.argv[2] if len(sys.argv) > 2 else "A"
# Modelos que no se entrenan esta vez, separados por comas (por ejemplo "EBM")
SALTAR = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
PERFILES = DATOS / "perfiles/opendata/profiles_opendata_normalized.pkl"
CARPETA = DATOS / "criba"
PARES = CARPETA / "pares_opendata.pkl"
RESULTADOS = CARPETA / f"resultados_{EMBEDDING}.csv"

# Repo de FREYJA: variable de entorno FREYJA_DIR, o por defecto una carpeta
# FREYJA junto a la de este proyecto
FREYJA = Path(os.environ.get("FREYJA_DIR", Path(__file__).resolve().parents[2] / "FREYJA"))
sys.path.insert(0, str(FREYJA))
from app.core.distances.compute_distances import (  # noqa: E402
    ComputeDistances, ComputeDistancesConfig)

SEMILLA = 0
PROPORCIONES = {"entrenamiento": 0.6, "validacion": 0.2, "test": 0.2}
CANDIDATAS = 100
MIN_CANDIDATAS = 5
K = 10
IDENTIFICADORES = ["consulta", "candidata", "parte"]
S = SEMILLA


# --------------------------------------------------------------------- datos

def columnas_con_embedding(modelo):
    columnas = pd.read_csv(DATOS / f"embeddings/opendata_{modelo}/columnas.csv")
    return {(t.removesuffix(".csv"), c): i
            for i, (t, c) in enumerate(zip(columnas["tabla"], columnas["columna_freyja"]))}


def cargar_perfiles():
    """Perfiles de las columnas que tienen embedding con los dos modelos."""
    calculadora = ComputeDistances(ComputeDistancesConfig(
        profiles_file_path=PERFILES, ground_truth_path=None,
        output_distances_path=CARPETA / "no_se_usa.csv"))
    con_los_dos = set(columnas_con_embedding("warpgate")) & set(columnas_con_embedding("mpnet"))
    claves = list(zip(calculadora.profiles["dataset_name"], calculadora.profiles["attribute_name"]))
    calculadora.profiles = calculadora.profiles[[c in con_los_dos for c in claves]].reset_index(drop=True)
    return calculadora


def repartir_tablas(perfiles):
    tablas = np.array(sorted(perfiles["dataset_name"].unique()))
    np.random.default_rng(SEMILLA).shuffle(tablas)
    cortes = (np.cumsum(list(PROPORCIONES.values())) * len(tablas)).round().astype(int)
    parte = {}
    for nombre, grupo in zip(PROPORCIONES, np.split(tablas, cortes[:-1])):
        parte.update({t: nombre for t in grupo})
    return perfiles["dataset_name"].map(parte).to_numpy()


def elegir_parejas(perfiles, parte_de_la_columna):
    """Para cada columna, con quién se empareja (siempre columnas de otra tabla de su parte)."""
    azar = np.random.default_rng(SEMILLA)
    tablas = perfiles["dataset_name"].to_numpy()
    parejas = set()
    for parte in PROPORCIONES:
        indices = np.flatnonzero(parte_de_la_columna == parte)
        for i in indices:
            otras = indices[tablas[indices] != tablas[i]]
            if parte != "test" and len(otras) > CANDIDATAS:
                otras = azar.choice(otras, CANDIDATAS, replace=False)
            parejas.update((i, j) for j in otras)
            parejas.update((j, i) for j in otras)  # los dos órdenes
    return parejas


def obtener_pares():
    if PARES.exists():
        return pd.read_pickle(PARES)
    calculadora = cargar_perfiles()
    perfiles = calculadora.profiles
    parte = repartir_tablas(perfiles)
    candidatas_de = {}
    for i, j in elegir_parejas(perfiles, parte):
        candidatas_de.setdefault(i, []).append(j)

    trozos = []
    for i, candidatas in sorted(candidatas_de.items()):
        candidatas = sorted(candidatas)
        distancias = calculadora._compute_distances_vectorized(
            perfiles.iloc[i], perfiles.iloc[candidatas].reset_index(drop=True))
        distancias.insert(0, "consulta", i)
        distancias.insert(1, "candidata", candidatas)
        distancias.insert(2, "parte", parte[i])
        trozos.append(distancias)
    pares = pd.concat(trozos, ignore_index=True)
    pares.attrs["claves"] = list(zip(perfiles["dataset_name"], perfiles["attribute_name"]))
    CARPETA.mkdir(parents=True, exist_ok=True)
    pares.to_pickle(PARES)
    return pares


def anadir_coseno(pares):
    fila = columnas_con_embedding(EMBEDDING)
    vectores = np.load(DATOS / f"embeddings/opendata_{EMBEDDING}/original.npy")
    unitarios = vectores / np.linalg.norm(vectores, axis=1, keepdims=True)
    filas = np.array([fila[c] for c in pares.attrs["claves"]])
    a = unitarios[filas[pares["consulta"].to_numpy()]]
    b = unitarios[filas[pares["candidata"].to_numpy()]]
    pares["coseno"] = (a * b).sum(axis=1)
    return pares


# ------------------------------------------------------------------ medidas

def spearman_por_consulta(pares, prediccion, objetivo="coseno"):
    """'objetivo' es la columna con el valor real (el coseno, salvo en src/objetivos.py)."""
    valores = []
    for _, grupo in pares.assign(p=prediccion).groupby("consulta"):
        if len(grupo) >= MIN_CANDIDATAS:
            rho = spearmanr(grupo[objetivo], grupo["p"]).correlation
            valores.append(0.0 if np.isnan(rho) else rho)
    return float(np.mean(valores))


def peso_en_el_top(valores, k):
    """Probabilidad de que cada candidata quede entre las k primeras si los empates
    se deshacen al azar. Las que están claramente dentro valen 1 y las de fuera 0.
    Las empatadas en la frontera se reparten las plazas que quedan: por ejemplo, 2
    plazas para 4 empatadas dan 0,5 a cada una."""
    k = min(k, len(valores))
    frontera = np.sort(valores)[::-1][k - 1]  # valor de la candidata en el puesto k
    dentro = valores > frontera
    empatadas = valores == frontera
    peso = dentro.astype(float)
    peso[empatadas] = (k - dentro.sum()) / empatadas.sum()
    return peso


def precision_con_empates(real, prediccion, k):
    """Precision@k esperada si los empates (en el coseno real y en la predicción) se
    deshacen al azar. Los dos desempates son independientes, así que la probabilidad
    de que una candidata esté en los dos top-k es el producto de sus dos pesos."""
    return float((peso_en_el_top(real, k) * peso_en_el_top(prediccion, k)).sum() / k)


def precision_en_k(pares, prediccion, objetivo="coseno"):
    """Precision@K media de las columnas consulta, con los empates deshechos al azar
    (versión esperada). La primera versión de la criba deshacía los empates por el
    orden de los datos, lo que daba resultados arbitrarios con modelos que empatan mucho."""
    aciertos = []
    for _, grupo in pares.assign(p=prediccion).groupby("consulta"):
        if len(grupo) >= MIN_CANDIDATAS:
            aciertos.append(precision_con_empates(grupo[objetivo].to_numpy(), grupo["p"].to_numpy(), K))
    return float(np.mean(aciertos))


def r2(real, prediccion, media):
    return 1 - ((real - prediccion) ** 2).sum() / ((real - media) ** 2).sum()


# ------------------------------------------------------------------ modelos

def boosting_xgb(**p):
    return XGBRegressor(n_estimators=5000, early_stopping_rounds=50, tree_method="hist",
                        n_jobs=-1, random_state=S, **p)


def boosting_lgbm(**p):
    return lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1, random_state=S, verbose=-1, **p)


CONFIGURACIONES = {
    "Media (referencia)": {"A": None},
    "GradientBoosting (FREYJA, referencia)": {
        "A": lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                                               subsample=0.8, min_samples_leaf=10, random_state=S)},
    "RandomForest": {
        "A": lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=S),
        "B": lambda: RandomForestRegressor(n_estimators=300, max_features=1 / 3, min_samples_leaf=5,
                                           n_jobs=-1, random_state=S),
        "C": lambda: RandomForestRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=20,
                                           max_samples=0.5, n_jobs=-1, random_state=S)},
    "ExtraTrees": {
        "A": lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=S),
        "B": lambda: ExtraTreesRegressor(n_estimators=300, max_features=1 / 3, min_samples_leaf=5,
                                         n_jobs=-1, random_state=S),
        "C": lambda: ExtraTreesRegressor(n_estimators=300, max_features=0.5, min_samples_leaf=20,
                                         bootstrap=True, max_samples=0.5, n_jobs=-1, random_state=S)},
    "HistGradientBoosting": {
        # No admite un conjunto de validación externo: para parar usa su propio 10 % al azar
        "A": lambda: HistGradientBoostingRegressor(max_iter=5000, early_stopping=True, random_state=S),
        "B": lambda: HistGradientBoostingRegressor(max_iter=5000, early_stopping=True, learning_rate=0.05,
                                                   max_leaf_nodes=15, min_samples_leaf=50, random_state=S),
        "C": lambda: HistGradientBoostingRegressor(max_iter=5000, early_stopping=True, learning_rate=0.05,
                                                   max_leaf_nodes=127, l2_regularization=1.0, random_state=S)},
    "XGBoost": {
        "A": lambda: boosting_xgb(),
        "B": lambda: boosting_xgb(learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8),
        "C": lambda: boosting_xgb(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=1.0)},
    "LightGBM": {
        "A": lambda: boosting_lgbm(),
        "B": lambda: boosting_lgbm(learning_rate=0.05, num_leaves=15, min_child_samples=50),
        "C": lambda: boosting_lgbm(learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_lambda=1.0)},
    "EBM": {
        "A": lambda: ExplainableBoostingRegressor(random_state=S, n_jobs=-1),
        "B": lambda: ExplainableBoostingRegressor(interactions=0, random_state=S, n_jobs=-1),
        "C": lambda: ExplainableBoostingRegressor(interactions=30, random_state=S, n_jobs=-1)},
    "RealMLP": {
        "A": lambda: RealMLP_TD_Regressor(random_state=S, device="cpu", verbosity=0),
        "B": lambda: RealMLP_TD_Regressor(random_state=S, device="cpu", verbosity=0, p_drop=0.3,
                                          wd=0.02),
        "C": lambda: RealMLP_TD_Regressor(random_state=S, device="cpu", verbosity=0,
                                          hidden_sizes=[512, 512, 512])},
    "KNeighbors (prueba rápida)": {
        "A": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance", n_jobs=-1)),
        "B": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(50, weights="distance", n_jobs=-1)),
        "C": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(200, weights="distance", n_jobs=-1))},
}


def entrenar(modelo, x_ent, y_ent, x_val, y_val):
    """Entrena; los modelos que lo admiten usan la validación para decidir cuándo parar."""
    if isinstance(modelo, XGBRegressor):
        modelo.fit(x_ent, y_ent, eval_set=[(x_val, y_val)], verbose=False)
        return modelo.best_iteration + 1
    if isinstance(modelo, lgb.LGBMRegressor):
        modelo.fit(x_ent, y_ent, eval_set=[(x_val, y_val)],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
        return modelo.best_iteration_
    if isinstance(modelo, RealMLP_TD_Regressor):
        modelo.fit(x_ent, y_ent, X_val=x_val, y_val=y_val)
        return None
    modelo.fit(x_ent, y_ent)
    return getattr(modelo, "n_iter_", None)


# --------------------------------------------------------------------- main

def main():
    pares = anadir_coseno(obtener_pares())
    variables = [c for c in pares.columns if c not in IDENTIFICADORES + ["coseno"]]
    partes = {p: pares[pares["parte"] == p] for p in PROPORCIONES}
    x = {p: d[variables].to_numpy(dtype=np.float32) for p, d in partes.items()}
    y = {p: d["coseno"].to_numpy() for p, d in partes.items()}
    media = y["entrenamiento"].mean()
    print(f"Embedding {EMBEDDING}. Pares: " + ", ".join(f"{p} {len(d)}" for p, d in partes.items())
          + f". Columnas consulta en test: {partes['test']['consulta'].nunique()}", flush=True)

    hechos = pd.read_csv(RESULTADOS) if RESULTADOS.exists() else pd.DataFrame(columns=["modelo", "config"])
    filas = hechos.to_dict("records")
    (CARPETA / "predicciones").mkdir(parents=True, exist_ok=True)

    for nombre, configuraciones in CONFIGURACIONES.items():
        if nombre in SALTAR:
            continue
        for config, construir in configuraciones.items():
            if ETAPA == "A" and config != "A":
                continue
            if ((hechos["modelo"] == nombre) & (hechos["config"] == config)).any():
                continue
            inicio = time.time()
            if construir is None:
                p = {parte: np.full(len(y[parte]), media) for parte in PROPORCIONES}
                arboles = None
            else:
                modelo = construir()
                arboles = entrenar(modelo, x["entrenamiento"], y["entrenamiento"],
                                   x["validacion"], y["validacion"])
                p = {parte: modelo.predict(x[parte]) for parte in PROPORCIONES}
            segundos = time.time() - inicio
            np.save(CARPETA / f"predicciones/{EMBEDDING}_{nombre.split(' ')[0]}_{config}.npy", p["test"])
            fila = {
                "modelo": nombre, "config": config, "segundos": round(segundos, 1), "arboles": arboles,
                "r2_entrenamiento": r2(y["entrenamiento"], p["entrenamiento"], media),
                "r2_validacion": r2(y["validacion"], p["validacion"], media),
                "spearman_validacion": spearman_por_consulta(partes["validacion"], p["validacion"]),
                "r2_test": r2(y["test"], p["test"], media),
                "spearman_test": spearman_por_consulta(partes["test"], p["test"]),
                "precision_10_test": precision_en_k(partes["test"], p["test"]),
                "mae_test": float(np.abs(y["test"] - p["test"]).mean()),
            }
            filas.append(fila)
            pd.DataFrame(filas).to_csv(RESULTADOS, index=False)
            print(f"{nombre} [{config}]: {segundos:.0f} s, R² val {fila['r2_validacion']:.3f}, "
                  f"Spearman val {fila['spearman_validacion']:.3f}", flush=True)

    print(f"\nEscrito en {RESULTADOS}")


if __name__ == "__main__":
    main()
