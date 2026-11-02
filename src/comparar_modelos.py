"""Exploración preliminar sobre FREYJABM: comparar varios modelos con un ajuste ligero de hiperparámetros.

Mismo experimento que experimento.py (v1): mismos pares, mismas 35 variables,
mismos 5 grupos de la validación cruzada y mismas medidas. Solo cambia el modelo.

Ajuste de hiperparámetros sin tocar el examen (validación anidada): en cada
ronda, uno de los 4 grupos de entrenamiento hace de validación. Cada
configuración se entrena con los otros 3 y se mide en ese grupo (Spearman por
consulta, la medida principal, D15). La mejor se vuelve a entrenar con los 4
grupos y solo entonces se evalúa en el examen.

El Gradient Boosting de FREYJA es la referencia: no se ajusta y se reutilizan sus
predicciones ya guardadas por experimento.py.

Uso (entorno .venv-ml):  python src/comparar_modelos.py [warpgate|mpnet]
Salida: data/experimento/modelos_freyjabm_<embedding>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

EMBEDDING = sys.argv[1] if len(sys.argv) > 1 else "warpgate"
sys.argv = [sys.argv[0], "v1", EMBEDDING]  # experimento.py lee la variante y el embedding
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experimento as e  # noqa: E402

SALIDA = e.DATOS / f"experimento/modelos_freyjabm_{EMBEDDING}.csv"
K = 10  # para Precision@10
S = e.SEMILLA

# Cada modelo: cómo se construye y las pocas configuraciones que se prueban
CANDIDATOS = {
    "Ridge (lineal)": (
        lambda p: make_pipeline(StandardScaler(), Ridge(**p)),
        [{"alpha": 1.0}, {"alpha": 100.0}]),
    "HistGradientBoosting": (
        lambda p: HistGradientBoostingRegressor(random_state=S, **p),
        [{"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300},
         {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 300},
         {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 500, "l2_regularization": 1.0}]),
    "Random Forest": (
        lambda p: RandomForestRegressor(n_estimators=200, max_samples=0.3, n_jobs=-1,
                                        random_state=S, **p),
        [{"max_features": 0.3, "min_samples_leaf": 10},
         {"max_features": 0.5, "min_samples_leaf": 50}]),
    "Extra Trees": (
        lambda p: ExtraTreesRegressor(n_estimators=200, bootstrap=True, max_samples=0.3,
                                      n_jobs=-1, random_state=S, **p),
        [{"max_features": 0.3, "min_samples_leaf": 10},
         {"max_features": 0.5, "min_samples_leaf": 50}]),
    "XGBoost": (
        lambda p: XGBRegressor(subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                               n_jobs=-1, random_state=S, **p),
        [{"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400},
         {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}]),
    "LightGBM": (
        lambda p: LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                n_jobs=-1, random_state=S, verbose=-1, **p),
        [{"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400},
         {"num_leaves": 63, "learning_rate": 0.1, "n_estimators": 300}]),
}


def preparar_pares():
    """Los mismos pares que experimento.py, con su cosine y su grupo."""
    perfiles, embeddings = e.cargar_datos()
    pares = e.obtener_distancias(perfiles)
    pares["coseno"] = e.calcular_cosenos(pares, embeddings)
    grupo_de_la_tabla = e.repartir_tablas(perfiles)
    grupo = [grupo_de_la_tabla[t] for t in perfiles["dataset_name"]]
    pares["grupo_consulta"] = [grupo[i] for i in pares["consulta"]]
    pares["grupo_candidata"] = [grupo[i] for i in pares["candidata"]]
    return pares


def precision_en_k(examen, prediccion):
    aciertos = []
    for _, sus_pares in examen.assign(prediccion=prediccion).groupby("consulta"):
        reales = set(sus_pares.nlargest(K, "coseno")["candidata"])
        predichas = set(sus_pares.nlargest(K, "prediccion")["candidata"])
        aciertos.append(len(reales & predichas) / K)
    return np.mean(aciertos)


def elegir_configuracion(construir, configuraciones, pares, variables, grupo_examen):
    """Elige la configuración con mejor Spearman por consulta en un grupo de validación."""
    if len(configuraciones) == 1:
        return configuraciones[0]
    grupo_validacion = (grupo_examen + 1) % e.GRUPOS
    fuera = [grupo_examen, grupo_validacion]
    validacion = ((pares["grupo_consulta"] == grupo_validacion)
                  & (pares["grupo_candidata"] == grupo_validacion))
    entrenamiento = (~pares["grupo_consulta"].isin(fuera) & ~pares["grupo_candidata"].isin(fuera))
    mejor, mejor_nota = None, -np.inf
    for configuracion in configuraciones:
        modelo = construir(configuracion)
        modelo.fit(pares.loc[entrenamiento, variables], pares.loc[entrenamiento, "coseno"])
        prediccion = modelo.predict(pares.loc[validacion, variables])
        nota = e.spearman_por_consulta(pares[validacion].assign(prediccion=prediccion))
        if nota > mejor_nota:
            mejor, mejor_nota = configuracion, nota
    return mejor


def main():
    pares = preparar_pares()
    variables = [c for c in pares.columns if c not in e.NO_SON_VARIABLES]
    referencia = pd.read_csv(e.PREDICCIONES)["prediccion"]  # Gradient Boosting de FREYJA

    filas = []
    for nombre, (construir, configuraciones) in {"Gradient Boosting (FREYJA)": (None, None),
                                                 **CANDIDATOS}.items():
        for grupo in range(e.GRUPOS):
            entrenamiento, examen = e.separar_ronda(pares, grupo)
            if construir is None:
                prediccion, configuracion = referencia[examen].to_numpy(), "ajustes de FREYJA"
            else:
                configuracion = elegir_configuracion(construir, configuraciones, pares,
                                                     variables, grupo)
                modelo = construir(configuracion)
                modelo.fit(pares.loc[entrenamiento, variables], pares.loc[entrenamiento, "coseno"])
                prediccion = modelo.predict(pares.loc[examen, variables])
            media = pares.loc[entrenamiento, "coseno"].mean()
            notas = e.evaluar_ronda(pares[examen].assign(prediccion=prediccion), media)
            notas.update({"modelo": nombre, "ronda": grupo, "configuracion": str(configuracion),
                          "precision_10": precision_en_k(pares[examen], prediccion)})
            filas.append(notas)
        print(f"{nombre}: hecho", flush=True)

    resultados = pd.DataFrame(filas)
    resultados.to_csv(SALIDA, index=False)
    medidas = ["r2", "spearman_por_consulta", "precision_10", "mae_modelo"]
    resumen = resultados.groupby("modelo", sort=False)[medidas].agg(["mean", "std"]).round(3)
    print(f"\nEmbedding {EMBEDDING}. Media (desviación típica) de las 5 rondas:")
    print(resumen.to_string())
    print("\nConfiguraciones elegidas en cada ronda:")
    print(resultados.pivot(index="modelo", columns="ronda", values="configuracion")
          .loc[list(CANDIDATOS)].to_string())
    print(f"\nEscrito en {SALIDA}")


if __name__ == "__main__":
    main()
