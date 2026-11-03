"""Análisis estadístico de la criba de modelos (OpenData).

El modelo se elige con la validación (regla D15): RandomForest B. El test no
decide nada, solo describe: notas con su incertidumbre y comparaciones.

Pasos:
  1. Nota de cada modelo en cada columna consulta del test, y por tabla.
     Precision@10 cuenta los empates como si se deshicieran al azar.
  2. Bootstrap emparejado por tablas: intervalos del 95 % de cada modelo y de
     las diferencias, más intervalos corregidos por Bonferroni (36 comparaciones).
     Sustituye al Wilcoxon previsto: responde a la misma pregunta.
  3. Curvas de aprendizaje de los tres principales: entrenar con el 10, 25 y 50 %
     de las tablas de entrenamiento (3 semillas) y medir en validación.

Uso (entorno .venv-ml):  python src/analisis.py <warpgate|mpnet> [curvas]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import criba  # noqa: E402  (mismos pares, mismo coseno y mismas medidas que la criba)

EMBEDDING = criba.EMBEDDING
CARPETA = criba.DATOS / "analisis"
NOTAS = CARPETA / f"notas_por_consulta_{EMBEDDING}.csv"

# Los modelos que se analizan: la mejor configuración de los tres principales
# según la validación, y la referencia de FREYJA. Clave: nombre corto del archivo
# de predicciones que guardó criba.py.
MODELOS = {
    "RandomForest_B": "RandomForest (B)",
    "LightGBM_C": "LightGBM (C)",
    "XGBoost_C": "XGBoost (C)",
    "GradientBoosting_A": "GradientBoosting FREYJA",
}


# ------------------------------------------------------------------ paso 1

def cargar_test():
    """Pares de test con su coseno, la media de entrenamiento y la tabla de cada consulta."""
    pares = criba.anadir_coseno(criba.obtener_pares())
    media = pares.loc[pares["parte"] == "entrenamiento", "coseno"].mean()
    test = pares[pares["parte"] == "test"]
    # La tabla de cada columna está en las claves guardadas con los pares
    tabla_de = [tabla for tabla, _ in pares.attrs["claves"]]
    test = test.assign(tabla=[tabla_de[i] for i in test["consulta"]])
    return test[["consulta", "candidata", "tabla", "coseno"]].reset_index(drop=True), media


def notas_de_una_consulta(real, prediccion, media):
    """Lo que hace falta de una columna consulta para todas las medidas."""
    rho = spearmanr(real, prediccion).correlation
    return {
        "spearman": 0.0 if np.isnan(rho) else rho,
        "precision_10": criba.precision_con_empates(real, prediccion, criba.K),
        # Para el R²: errores al cuadrado del modelo y de predecir la media.
        # Se guardan las sumas (no el R² de cada consulta) para poder juntarlas después.
        "error_modelo": float(((real - prediccion) ** 2).sum()),
        "error_media": float(((real - media) ** 2).sum()),
    }


def calcular_notas(test, media):
    """Una fila por modelo y columna consulta."""
    filas = []
    for archivo, nombre in MODELOS.items():
        prediccion = np.load(criba.CARPETA / f"predicciones/{EMBEDDING}_{archivo}.npy")
        assert len(prediccion) == len(test), "las predicciones no corresponden a estos pares"
        for consulta, grupo in test.assign(prediccion=prediccion).groupby("consulta"):
            if len(grupo) < criba.MIN_CANDIDATAS:
                continue
            notas = notas_de_una_consulta(grupo["coseno"].to_numpy(), grupo["prediccion"].to_numpy(),
                                          media)
            filas.append({"modelo": nombre, "consulta": consulta, "tabla": grupo["tabla"].iloc[0],
                          "candidatas": len(grupo), **notas})
        print(f"  {nombre}: hecho", flush=True)
    return pd.DataFrame(filas)


def obtener_notas():
    if NOTAS.exists():
        return pd.read_csv(NOTAS)
    test, media = cargar_test()
    notas = calcular_notas(test, media)
    CARPETA.mkdir(parents=True, exist_ok=True)
    notas.to_csv(NOTAS, index=False)
    return notas


def comprobar_con_la_criba(notas):
    """Las medias deben coincidir con las notas de test que guardó criba.py."""
    criba_guardada = pd.read_csv(criba.RESULTADOS)
    print("\nComprobación con los resultados de la criba (test):")
    for archivo, nombre in MODELOS.items():
        modelo, config = archivo.split("_")
        fila = criba_guardada[criba_guardada["modelo"].str.startswith(modelo)
                              & (criba_guardada["config"] == config)].iloc[0]
        suyas = notas[notas["modelo"] == nombre]
        r2 = 1 - suyas["error_modelo"].sum() / suyas["error_media"].sum()
        print(f"  {nombre:24s} Spearman {suyas['spearman'].mean():.4f} (criba {fila['spearman_test']:.4f})"
              f"  P@10 {suyas['precision_10'].mean():.4f} (criba {fila['precision_10_test']:.4f})"
              f"  R² {r2:.4f} (criba {fila['r2_test']:.4f})")


def resumen_por_tabla(notas):
    """Nota media de cada modelo en cada tabla (la unidad independiente del test)."""
    return notas.groupby(["modelo", "tabla"], as_index=False).agg(
        consultas=("consulta", "size"), spearman=("spearman", "mean"),
        precision_10=("precision_10", "mean"),
        error_modelo=("error_modelo", "sum"), error_media=("error_media", "sum"))


# ------------------------------------------------------------------ paso 2

REPETICIONES = 20000
MEDIDAS = ["spearman", "precision_10", "r2"]
# Corrección de Bonferroni: se presentan 6 parejas de modelos × 3 medidas × 2
# embeddings = 36 comparaciones. Para que el conjunto tenga un 95 % de fiabilidad,
# cada intervalo de una diferencia se calcula al 1 - 0,05 / 36 (99,86 %).
COMPARACIONES = 36
ALFA_CORREGIDO = 0.05 / COMPARACIONES


def sumas_por_tabla(notas):
    """Para cada modelo, una fila por tabla (en el mismo orden) con las sumas que
    permiten recalcular cualquier medida sobre un conjunto de tablas."""
    sumas = notas.groupby(["modelo", "tabla"]).agg(
        consultas=("consulta", "size"), spearman=("spearman", "sum"),
        precision_10=("precision_10", "sum"),
        error_modelo=("error_modelo", "sum"), error_media=("error_media", "sum"))
    return {nombre: sumas.loc[nombre].sort_index() for nombre in MODELOS.values()}


def medidas(sumas, pesos):
    """Las tres medidas con cada tabla contada tantas veces como diga 'pesos'.
    'pesos' tiene una fila por repetición del bootstrap y una columna por tabla."""
    return {
        "spearman": pesos @ sumas["spearman"].to_numpy() / (pesos @ sumas["consultas"].to_numpy()),
        "precision_10": pesos @ sumas["precision_10"].to_numpy() / (pesos @ sumas["consultas"].to_numpy()),
        "r2": 1 - pesos @ sumas["error_modelo"].to_numpy() / (pesos @ sumas["error_media"].to_numpy()),
    }


def bootstrap(notas):
    """Intervalos del 95 % de cada modelo y de la diferencia entre cada par de modelos.
    En cada repetición se eligen al azar, con reemplazo, tantas tablas como hay en el
    test. Los pesos cuentan cuántas veces sale cada tabla. Todas las medidas y todos
    los modelos de una repetición usan los mismos pesos (comparación emparejada)."""
    sumas = sumas_por_tabla(notas)
    tablas = len(next(iter(sumas.values())))
    azar = np.random.default_rng(criba.SEMILLA)
    elegidas = azar.integers(0, tablas, size=(REPETICIONES, tablas))
    pesos = np.zeros((REPETICIONES, tablas))
    np.add.at(pesos, (np.arange(REPETICIONES)[:, None], elegidas), 1)
    todas = np.ones((1, tablas))  # el test tal cual, para la nota puntual

    nota, repeticion = {}, {}
    for nombre, suyas in sumas.items():
        nota[nombre] = {m: v[0] for m, v in medidas(suyas, todas).items()}
        repeticion[nombre] = medidas(suyas, pesos)

    filas = []
    for nombre in sumas:
        for m in MEDIDAS:
            bajo, alto = np.percentile(repeticion[nombre][m], [2.5, 97.5])
            filas.append({"comparacion": nombre, "medida": m, "nota": nota[nombre][m],
                          "bajo": bajo, "alto": alto})
    nombres = list(sumas)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            for m in MEDIDAS:
                diferencia = repeticion[a][m] - repeticion[b][m]
                bajo, alto = np.percentile(diferencia, [2.5, 97.5])
                bajo_c, alto_c = np.percentile(diferencia, [50 * ALFA_CORREGIDO, 100 - 50 * ALFA_CORREGIDO])
                filas.append({"comparacion": f"{a} − {b}", "medida": m,
                              "nota": nota[a][m] - nota[b][m], "bajo": bajo, "alto": alto,
                              "bajo_corregido": bajo_c, "alto_corregido": alto_c,
                              # proporción de repeticiones en que gana a
                              "gana_a": float((diferencia > 0).mean())})
    return pd.DataFrame(filas)


def mostrar_bootstrap(resultado):
    for m in MEDIDAS:
        print(f"\n{m}  (nota [intervalo del 95 %])")
        for _, fila in resultado[resultado["medida"] == m].iterrows():
            extra = ""
            if not np.isnan(fila.get("gana_a", np.nan)):
                fiable = "FIABLE" if not fila["bajo_corregido"] <= 0 <= fila["alto_corregido"] else "no fiable"
                extra = (f"   corregido [{fila['bajo_corregido']:+.4f}, {fila['alto_corregido']:+.4f}] {fiable},"
                         f" gana el primero en el {fila['gana_a']:.1%}")
            print(f"  {fila['comparacion']:45s} {fila['nota']:+.4f} [{fila['bajo']:+.4f}, {fila['alto']:+.4f}]{extra}")


# ------------------------------------------------------------------ paso 3

FRACCIONES = [0.1, 0.25, 0.5]
SEMILLAS_CURVAS = [0, 1, 2]
CURVAS = CARPETA / f"curvas_{EMBEDDING}.csv"
PRINCIPALES = {"RandomForest (B)": ("RandomForest", "B"), "LightGBM (C)": ("LightGBM", "C"),
               "XGBoost (C)": ("XGBoost", "C")}


def curvas_de_aprendizaje():
    """Entrena los tres principales con una parte de las tablas de entrenamiento.
    Se eligen al azar las tablas y se conservan los pares cuya columna consulta es
    de esas tablas: cada columna mantiene sus 100 candidatas y el número de pares
    baja en la misma proporción. La validación es siempre la misma. El punto del
    100 % es el de la criba."""
    import time
    pares = criba.anadir_coseno(criba.obtener_pares())
    variables = [c for c in pares.columns if c not in criba.IDENTIFICADORES + ["coseno"]]
    tabla_de = np.array([tabla for tabla, _ in pares.attrs["claves"]])
    entrenamiento = pares[pares["parte"] == "entrenamiento"]
    validacion = pares[pares["parte"] == "validacion"]
    x_val = validacion[variables].to_numpy(dtype=np.float32)
    y_val = validacion["coseno"].to_numpy()
    media = entrenamiento["coseno"].mean()  # la misma referencia que la criba
    tabla_consulta = tabla_de[entrenamiento["consulta"].to_numpy()]
    tablas = np.array(sorted(set(tabla_consulta)))

    hechos = pd.read_csv(CURVAS) if CURVAS.exists() else pd.DataFrame(
        columns=["modelo", "fraccion", "semilla"])
    filas = hechos.to_dict("records")
    for fraccion in FRACCIONES:
        for semilla in SEMILLAS_CURVAS:
            elegidas = np.random.default_rng(semilla).choice(
                tablas, round(fraccion * len(tablas)), replace=False)
            parte = entrenamiento[np.isin(tabla_consulta, elegidas)]
            x = parte[variables].to_numpy(dtype=np.float32)
            y = parte["coseno"].to_numpy()
            for nombre, (modelo, config) in PRINCIPALES.items():
                if ((hechos["modelo"] == nombre) & (hechos["fraccion"] == fraccion)
                        & (hechos["semilla"] == semilla)).any():
                    continue
                inicio = time.time()
                estimador = criba.CONFIGURACIONES[modelo][config]()
                criba.entrenar(estimador, x, y, x_val, y_val)
                p_val = estimador.predict(x_val)
                filas.append({
                    "modelo": nombre, "fraccion": fraccion, "semilla": semilla,
                    "tablas": len(elegidas), "pares": len(parte),
                    "segundos": round(time.time() - inicio, 1),
                    "r2_entrenamiento": criba.r2(y, estimador.predict(x), media),
                    "r2_validacion": criba.r2(y_val, p_val, media),
                    "spearman_validacion": criba.spearman_por_consulta(validacion, p_val)})
                pd.DataFrame(filas).to_csv(CURVAS, index=False)
                print(f"  {nombre} {fraccion:.0%} semilla {semilla}: "
                      f"Spearman val {filas[-1]['spearman_validacion']:.3f} "
                      f"({filas[-1]['segundos']:.0f} s)", flush=True)

    # El punto del 100 % sale de la criba (todas las tablas, una sola ejecución)
    criba_guardada = pd.read_csv(criba.RESULTADOS)
    for nombre, (modelo, config) in PRINCIPALES.items():
        fila = criba_guardada[(criba_guardada["modelo"] == modelo) & (criba_guardada["config"] == config)].iloc[0]
        filas.append({"modelo": nombre, "fraccion": 1.0, "semilla": 0, "tablas": len(tablas),
                      "pares": len(entrenamiento), "segundos": fila["segundos"],
                      "r2_entrenamiento": fila["r2_entrenamiento"], "r2_validacion": fila["r2_validacion"],
                      "spearman_validacion": fila["spearman_validacion"]})
    curvas = pd.DataFrame(filas).drop_duplicates(["modelo", "fraccion", "semilla"], keep="last")
    curvas.to_csv(CURVAS, index=False)
    resumen = curvas.groupby(["modelo", "fraccion"]).agg(
        pares=("pares", "mean"), spearman=("spearman_validacion", "mean"),
        spearman_sd=("spearman_validacion", "std"), r2_val=("r2_validacion", "mean"),
        r2_ent=("r2_entrenamiento", "mean")).round(4)
    print(resumen.to_string())


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "curvas":
        curvas_de_aprendizaje()
        return
    notas = obtener_notas()
    comprobar_con_la_criba(notas)

    por_tabla = resumen_por_tabla(notas)
    tablas = por_tabla["tabla"].nunique()
    consultas = notas["consulta"].nunique()
    print(f"\nTest: {consultas} columnas consulta en {tablas} tablas")
    print("Columnas consulta por tabla:",
          por_tabla.drop_duplicates("tabla")["consultas"].describe()[["min", "50%", "mean", "max"]]
          .round(1).to_dict())

    resultado = bootstrap(notas)
    resultado.to_csv(CARPETA / f"bootstrap_{EMBEDDING}.csv", index=False)
    print(f"\nBootstrap por tablas, {REPETICIONES} repeticiones")
    mostrar_bootstrap(resultado)


if __name__ == "__main__":
    main()
