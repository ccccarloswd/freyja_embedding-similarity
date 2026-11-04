"""Punto 2: predecir el producto escalar y la distancia euclídea en lugar del coseno (D37).

Mismo montaje que la criba: mismos pares, mismo reparto, mismas medidas y
RandomForest B, el modelo elegido (D36). Solo cambia el objetivo, calculado con
los vectores SIN normalizar (con vectores de longitud 1 los tres objetivos
ordenarían igual):
  - WarpGate: original.npy (TaBERT no normaliza).
  - MPNet: sin_normalizar.npy (el modelo normaliza en su último paso, aquí no).

La distancia euclídea se usa con signo negativo: así, como en el coseno y en el
producto escalar, un valor más alto significa "más parecido", y todas las medidas
(Spearman, Precision@10, R²) se calculan igual.

Pasos:
  1. Entrenar RandomForest B con el nuevo objetivo y medirlo.
  2. Referencia: el modelo del coseno (ya entrenado en la criba) ordenando según
     el nuevo objetivo, frente al modelo nuevo. Solo medidas de orden (Spearman y
     Precision@10): el R² no sirve porque las escalas son distintas. Bootstrap
     emparejado por tablas, como en src/analisis.py.
  3. Comparar entre sí las predicciones de los dos modelos, sin mirar el objetivo
     real: ¿ordenan igual o han aprendido cosas distintas?
  4. Diagnóstico: predecir la norma de cada columna con su perfil (sus valores).
  5. Producto por partes: coseno predicho × norma predicha de a × norma predicha de b.
  6. D38: el mismo modelo con 95 variables (las 35 comparaciones más las 30 métricas
     numéricas propias de cada columna del par), para el coseno y el producto escalar.

Uso (entorno .venv-ml):
  python src/objetivos.py <warpgate|mpnet> <producto|euclidea>            (paso 1)
  python src/objetivos.py <warpgate|mpnet> <producto|euclidea> comparar   (pasos 2 y 3)
  python src/objetivos.py <warpgate|mpnet> norma                          (paso 4)
  python src/objetivos.py <warpgate|mpnet> producto partes                (paso 5)
  python src/objetivos.py <warpgate|mpnet> <coseno|producto> ampliado     (paso 6, entrenar)
  python src/objetivos.py <warpgate|mpnet> <coseno|producto> comparar_ampliado
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import criba  # noqa: E402  (mismos pares, mismo reparto, mismo modelo y mismas medidas)

EMBEDDING = criba.EMBEDDING
OBJETIVO = sys.argv[2] if len(sys.argv) > 2 else "producto"
PASO = sys.argv[3] if len(sys.argv) > 3 else "entrenar"
AMPLIADO = PASO == "ampliado"  # D38: 95 variables en lugar de 35
NOMBRE = f"{OBJETIVO}_ampliado" if AMPLIADO else OBJETIVO
MODELO = sys.argv[4] if len(sys.argv) > 4 else "RandomForest_B"  # familia_configuración de la criba
VECTORES = {"warpgate": "original.npy", "mpnet": "sin_normalizar.npy"}
CARPETA = criba.DATOS / "objetivos"
RESULTADOS = CARPETA / "resultados.csv"
BLOQUE = 200_000  # pares por bloque al calcular el objetivo


# ------------------------------------------------------------------ paso 1

def anadir_objetivo(pares):
    """Añade la columna 'objetivo' (coseno, producto escalar o menos la distancia euclídea)."""
    if OBJETIVO == "coseno":  # el mismo coseno que la criba
        pares = criba.anadir_coseno(pares)
        pares["objetivo"] = pares.pop("coseno")
        return pares
    fila = criba.columnas_con_embedding(EMBEDDING)
    vectores = np.load(criba.DATOS / f"embeddings/opendata_{EMBEDDING}/{VECTORES[EMBEDDING]}")
    filas = np.array([fila[c] for c in pares.attrs["claves"]])
    consulta = filas[pares["consulta"].to_numpy()]
    candidata = filas[pares["candidata"].to_numpy()]
    # Por bloques de pares: con todos a la vez, WarpGate ocuparía unos 20-30 GB de memoria
    objetivo = np.empty(len(pares))
    for inicio in range(0, len(pares), BLOQUE):
        a = vectores[consulta[inicio:inicio + BLOQUE]]
        b = vectores[candidata[inicio:inicio + BLOQUE]]
        if OBJETIVO == "producto":
            objetivo[inicio:inicio + BLOQUE] = (a * b).sum(axis=1)
        else:
            objetivo[inicio:inicio + BLOQUE] = -np.linalg.norm(a - b, axis=1)
    pares["objetivo"] = objetivo
    return pares


def perfiles_numericos():
    """Las métricas numéricas del perfil de cada columna, en el mismo orden que los
    índices de los pares. Las de texto (palabras frecuentes, primera y última
    palabra) se dejan fuera: solo tienen sentido al comparar dos columnas."""
    perfiles = criba.cargar_perfiles().profiles
    variables = [c for c in perfiles.columns
                 if c not in ("dataset_name", "attribute_name") and perfiles[c].dtype != object]
    return perfiles, perfiles[variables].to_numpy(dtype=np.float32), variables


def entrenar_y_medir():
    pares = anadir_objetivo(criba.obtener_pares())
    variables = [c for c in pares.columns if c not in criba.IDENTIFICADORES + ["objetivo"]]
    partes = {p: pares[pares["parte"] == p] for p in criba.PROPORCIONES}
    x = {p: d[variables].to_numpy(dtype=np.float32) for p, d in partes.items()}
    if AMPLIADO:
        # D38: además de las 35 comparaciones, los valores propios de las dos columnas
        _, propios, _ = perfiles_numericos()
        assert len(propios) == len(pares.attrs["claves"])
        x = {p: np.hstack([x[p], propios[d["consulta"].to_numpy()], propios[d["candidata"].to_numpy()]])
             for p, d in partes.items()}
    print(f"{EMBEDDING}, {NOMBRE}: {x['entrenamiento'].shape[1]} variables", flush=True)
    y = {p: d["objetivo"].to_numpy() for p, d in partes.items()}
    media = y["entrenamiento"].mean()

    inicio = time.time()
    familia, config = MODELO.split("_")
    modelo = criba.CONFIGURACIONES[familia][config]()
    # Igual que en la criba: LightGBM y XGBoost usan la validación para decidir cuándo parar
    arboles = criba.entrenar(modelo, x["entrenamiento"], y["entrenamiento"], x["validacion"], y["validacion"])
    p = {parte: modelo.predict(x[parte]) for parte in criba.PROPORCIONES}
    segundos = time.time() - inicio

    (CARPETA / "predicciones").mkdir(parents=True, exist_ok=True)
    np.save(CARPETA / f"predicciones/{EMBEDDING}_{NOMBRE}_{MODELO}.npy", p["test"])
    return {
        "embedding": EMBEDDING, "objetivo": NOMBRE, "modelo": MODELO, "arboles": arboles,
        "segundos": round(segundos, 1),
        "r2_entrenamiento": criba.r2(y["entrenamiento"], p["entrenamiento"], media),
        "r2_validacion": criba.r2(y["validacion"], p["validacion"], media),
        "spearman_validacion": criba.spearman_por_consulta(partes["validacion"], p["validacion"], "objetivo"),
        "r2_test": criba.r2(y["test"], p["test"], media),
        "spearman_test": criba.spearman_por_consulta(partes["test"], p["test"], "objetivo"),
        "precision_10_test": criba.precision_en_k(partes["test"], p["test"], "objetivo"),
    }


def main():
    fila = entrenar_y_medir()
    anteriores = (pd.read_csv(RESULTADOS) if RESULTADOS.exists()
                  else pd.DataFrame(columns=["embedding", "objetivo", "modelo"]))
    if "modelo" not in anteriores:  # las primeras filas eran todas de RandomForest B
        anteriores.insert(2, "modelo", "RandomForest_B")
    anteriores = anteriores[~((anteriores["embedding"] == EMBEDDING) & (anteriores["objetivo"] == NOMBRE)
                              & (anteriores["modelo"] == MODELO))]
    resultados = pd.concat([anteriores, pd.DataFrame([fila])], ignore_index=True)
    resultados.to_csv(RESULTADOS, index=False)
    print(pd.DataFrame([fila]).round(4).to_string(index=False))


# ------------------------------------------------------------------ pasos 2 y 3

REPETICIONES = 20000
# Corrección de Bonferroni: 2 medidas × 2 objetivos × 2 embeddings = 8 comparaciones
COMPARACIONES = 8
ALFA_CORREGIDO = 0.05 / COMPARACIONES


def nota_de_orden(real, prediccion):
    rho = spearmanr(real, prediccion).correlation
    return 0.0 if np.isnan(rho) else rho, criba.precision_con_empates(real, prediccion, criba.K)


def notas_por_consulta():
    """Una fila por columna consulta del test con:
    - paso 2: Spearman y Precision@10 de cada modelo frente al objetivo real;
    - paso 3: Spearman y top-10 compartido entre las predicciones de los dos modelos."""
    pares = anadir_objetivo(criba.obtener_pares())
    tabla_de = [tabla for tabla, _ in pares.attrs["claves"]]
    test = pares.loc[pares["parte"] == "test", ["consulta", "objetivo"]].reset_index(drop=True)
    del pares
    coseno = np.load(criba.CARPETA / f"predicciones/{EMBEDDING}_RandomForest_B.npy")
    nuevo = np.load(CARPETA / f"predicciones/{EMBEDDING}_{OBJETIVO}_RandomForest_B.npy")
    assert len(coseno) == len(nuevo) == len(test), "las predicciones no corresponden a estos pares"

    filas = []
    for consulta, g in test.assign(coseno=coseno, nuevo=nuevo).groupby("consulta"):
        if len(g) < criba.MIN_CANDIDATAS:
            continue
        real, p_coseno, p_nuevo = g["objetivo"].to_numpy(), g["coseno"].to_numpy(), g["nuevo"].to_numpy()
        fila = {"consulta": consulta, "tabla": tabla_de[consulta], "candidatas": len(g)}
        fila["spearman_coseno"], fila["precision_10_coseno"] = nota_de_orden(real, p_coseno)
        fila["spearman_nuevo"], fila["precision_10_nuevo"] = nota_de_orden(real, p_nuevo)
        fila["spearman_entre_modelos"], fila["top_10_compartido"] = nota_de_orden(p_coseno, p_nuevo)
        filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_paso_2(notas):
    """Diferencia modelo nuevo − modelo del coseno, remuestreando tablas del test.
    Mismos pesos para los dos modelos (comparación emparejada)."""
    por_tabla = notas.groupby("tabla").agg(
        consultas=("consulta", "size"),
        **{c: (c, "sum") for c in ["spearman_coseno", "spearman_nuevo",
                                   "precision_10_coseno", "precision_10_nuevo"]})
    tablas = len(por_tabla)
    elegidas = np.random.default_rng(criba.SEMILLA).integers(0, tablas, size=(REPETICIONES, tablas))
    pesos = np.zeros((REPETICIONES, tablas))
    np.add.at(pesos, (np.arange(REPETICIONES)[:, None], elegidas), 1)
    n = pesos @ por_tabla["consultas"].to_numpy()

    filas = []
    for medida in ("spearman", "precision_10"):
        coseno = pesos @ por_tabla[f"{medida}_coseno"].to_numpy() / n
        nuevo = pesos @ por_tabla[f"{medida}_nuevo"].to_numpy() / n
        diferencia = nuevo - coseno
        bajo, alto = np.percentile(diferencia, [2.5, 97.5])
        bajo_c, alto_c = np.percentile(diferencia, [50 * ALFA_CORREGIDO, 100 - 50 * ALFA_CORREGIDO])
        filas.append({
            "embedding": EMBEDDING, "objetivo": OBJETIVO, "medida": medida,
            "modelo_coseno": notas[f"{medida}_coseno"].mean(),
            "modelo_nuevo": notas[f"{medida}_nuevo"].mean(),
            "diferencia": notas[f"{medida}_nuevo"].mean() - notas[f"{medida}_coseno"].mean(),
            "bajo": bajo, "alto": alto, "bajo_corregido": bajo_c, "alto_corregido": alto_c,
            "gana_nuevo": float((diferencia > 0).mean())})
    return pd.DataFrame(filas)


def comparar():
    notas = notas_por_consulta()
    notas.to_csv(CARPETA / f"notas_{EMBEDDING}_{OBJETIVO}.csv", index=False)
    guardado = pd.read_csv(RESULTADOS)
    guardado = guardado[(guardado["embedding"] == EMBEDDING) & (guardado["objetivo"] == OBJETIVO)].iloc[0]
    print(f"{EMBEDDING}, {OBJETIVO}: {len(notas)} columnas consulta en {notas['tabla'].nunique()} tablas")
    print(f"Comprobación con el paso 1: Spearman {notas['spearman_nuevo'].mean():.4f} "
          f"(paso 1 {guardado['spearman_test']:.4f}), P@10 {notas['precision_10_nuevo'].mean():.4f} "
          f"(paso 1 {guardado['precision_10_test']:.4f})")

    resultado = bootstrap_paso_2(notas)
    resultado.to_csv(CARPETA / f"bootstrap_{EMBEDDING}_{OBJETIVO}.csv", index=False)
    print()
    print(f"Paso 2: ordenar por el {OBJETIVO} real (bootstrap por tablas, {REPETICIONES} repeticiones)")
    for _, f in resultado.iterrows():
        fiable = "FIABLE" if not f["bajo_corregido"] <= 0 <= f["alto_corregido"] else "no fiable"
        print(f"  {f['medida']:13s} coseno {f['modelo_coseno']:.4f}  nuevo {f['modelo_nuevo']:.4f}  "
              f"nuevo - coseno {f['diferencia']:+.4f} [{f['bajo']:+.4f}, {f['alto']:+.4f}]  "
              f"corregido [{f['bajo_corregido']:+.4f}, {f['alto_corregido']:+.4f}] {fiable}, "
              f"gana el nuevo en el {f['gana_nuevo']:.1%}")

    print()
    print("Paso 3: los dos modelos entre sí (por columna consulta)")
    for c in ("spearman_entre_modelos", "top_10_compartido"):
        q = notas[c].quantile([0.1, 0.5, 0.9])
        print(f"  {c:23s} media {notas[c].mean():.4f}  percentiles 10/50/90: "
              f"{q[0.1]:.3f} / {q[0.5]:.3f} / {q[0.9]:.3f}")


# ------------------------------------------------------------------ paso 4

def norma():
    """¿Saben los perfiles la norma? Una fila por columna (no por par): como entrada,
    las métricas numéricas de su perfil (sus valores, no diferencias), y como
    objetivo, la norma de su vector. Las métricas de texto (palabras frecuentes,
    primera y última palabra) se dejan fuera: solo tienen sentido al comparar dos
    columnas. Mismo reparto por tablas y mismo modelo que la criba."""
    from sklearn.inspection import permutation_importance

    perfiles, x, variables = perfiles_numericos()
    parte = criba.repartir_tablas(perfiles)
    fila = criba.columnas_con_embedding(EMBEDDING)
    vectores = np.load(criba.DATOS / f"embeddings/opendata_{EMBEDDING}/{VECTORES[EMBEDDING]}")
    filas = [fila[c] for c in zip(perfiles["dataset_name"], perfiles["attribute_name"])]
    y = np.linalg.norm(vectores[filas], axis=1)
    x_de = {p: x[parte == p] for p in criba.PROPORCIONES}
    y_de = {p: y[parte == p] for p in criba.PROPORCIONES}
    media = y_de["entrenamiento"].mean()

    modelo = criba.CONFIGURACIONES["RandomForest"]["B"]()
    modelo.fit(x_de["entrenamiento"], y_de["entrenamiento"])
    p = {parte_: modelo.predict(x_de[parte_]) for parte_ in criba.PROPORCIONES}
    # Norma predicha de todas las columnas, en el orden de los pares (para el producto
    # por partes solo se usan las de test, que el modelo no ha visto)
    (CARPETA / "predicciones").mkdir(parents=True, exist_ok=True)
    np.save(CARPETA / f"predicciones/{EMBEDDING}_norma_RandomForest_B.npy", modelo.predict(x))

    print(f"{EMBEDDING}: {len(y)} columnas, {len(variables)} métricas del perfil")
    print(f"Norma: media {y.mean():.3f}, desviación {y.std():.3f} (variación {y.std() / y.mean():.1%})")
    resultado = {"embedding": EMBEDDING, "columnas": len(y), "norma_media": y.mean(),
                 "norma_variacion": y.std() / y.mean()}
    for parte_ in criba.PROPORCIONES:
        resultado[f"r2_{parte_}"] = criba.r2(y_de[parte_], p[parte_], media)
        resultado[f"spearman_{parte_}"] = spearmanr(y_de[parte_], p[parte_]).correlation
        print(f"  {parte_:13s} {len(y_de[parte_]):5d} columnas  R² {resultado[f'r2_{parte_}']:.3f}"
              f"  Spearman {resultado[f'spearman_{parte_}']:.3f}")

    # Importancia por permutación en validación: cuánto baja el R² al desordenar cada métrica
    importancia = permutation_importance(modelo, x_de["validacion"], y_de["validacion"],
                                         n_repeats=10, random_state=criba.SEMILLA, n_jobs=-1)
    tabla = pd.DataFrame({"metrica": variables, "importancia": importancia.importances_mean,
                          "desviacion": importancia.importances_std}).sort_values("importancia", ascending=False)
    tabla.to_csv(CARPETA / f"norma_importancias_{EMBEDDING}.csv", index=False)
    print("Métricas más importantes (bajada del R² de validación al desordenarlas):")
    print(tabla.head(8).round(4).to_string(index=False))

    archivo = CARPETA / "norma.csv"
    anteriores = pd.read_csv(archivo) if archivo.exists() else pd.DataFrame(columns=["embedding"])
    anteriores = anteriores[anteriores["embedding"] != EMBEDDING]
    pd.concat([anteriores, pd.DataFrame([resultado])], ignore_index=True).to_csv(archivo, index=False)


# ------------------------------------------- comparación general (producto por partes y D38)

def test_con_objetivo():
    """Pares de test con su objetivo real y su tabla, y la media de entrenamiento (para el R²)."""
    pares = anadir_objetivo(criba.obtener_pares())
    tabla_de = [tabla for tabla, _ in pares.attrs["claves"]]
    media = pares.loc[pares["parte"] == "entrenamiento", "objetivo"].mean()
    test = pares.loc[pares["parte"] == "test", ["consulta", "candidata", "objetivo"]].reset_index(drop=True)
    return test.assign(tabla=[tabla_de[i] for i in test["consulta"]]), media


def notas_de_modelos(test, predicciones, media):
    """Una fila por columna consulta, con las notas de cada modelo frente al objetivo real.
    Para el R² se guardan las sumas de errores, que se pueden juntar por tablas."""
    filas = []
    columnas = {n: f"p_{n}" for n in predicciones}
    datos = test.assign(**{columnas[n]: p for n, p in predicciones.items()})
    for consulta, g in datos.groupby("consulta"):
        if len(g) < criba.MIN_CANDIDATAS:
            continue
        real = g["objetivo"].to_numpy()
        fila = {"consulta": consulta, "tabla": g["tabla"].iloc[0], "error_media": float(((real - media) ** 2).sum())}
        for n, c in columnas.items():
            p = g[c].to_numpy()
            fila[f"spearman_{n}"], fila[f"precision_10_{n}"] = nota_de_orden(real, p)
            fila[f"error_{n}"] = float(((real - p) ** 2).sum())
        filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_parejas(notas, parejas, medidas, comparaciones):
    """Diferencia a − b de cada pareja de modelos, remuestreando tablas del test con los
    mismos pesos para todos (comparación emparejada). Bonferroni con 'comparaciones'."""
    alfa = 0.05 / comparaciones
    suma = notas.groupby("tabla").sum(numeric_only=True)
    n_consultas = notas.groupby("tabla").size().to_numpy()
    tablas = len(suma)
    elegidas = np.random.default_rng(criba.SEMILLA).integers(0, tablas, size=(REPETICIONES, tablas))
    pesos = np.zeros((REPETICIONES, tablas))
    np.add.at(pesos, (np.arange(REPETICIONES)[:, None], elegidas), 1)
    todas = np.ones((1, tablas))

    def medida(w, m, modelo):
        if m == "r2":
            return 1 - w @ suma[f"error_{modelo}"].to_numpy() / (w @ suma["error_media"].to_numpy())
        return w @ suma[f"{m}_{modelo}"].to_numpy() / (w @ n_consultas)

    filas = []
    for a, b in parejas:
        for m in medidas:
            diferencia = medida(pesos, m, a) - medida(pesos, m, b)
            bajo_c, alto_c = np.percentile(diferencia, [50 * alfa, 100 - 50 * alfa])
            filas.append({"embedding": EMBEDDING, "objetivo": OBJETIVO, "a": a, "b": b, "medida": m,
                          "nota_a": medida(todas, m, a)[0], "nota_b": medida(todas, m, b)[0],
                          "diferencia": medida(todas, m, a)[0] - medida(todas, m, b)[0],
                          "bajo_corregido": bajo_c, "alto_corregido": alto_c,
                          "gana_a": float((diferencia > 0).mean())})
    return pd.DataFrame(filas)


def mostrar(resultado):
    for _, f in resultado.iterrows():
        fiable = "FIABLE" if not f["bajo_corregido"] <= 0 <= f["alto_corregido"] else "no fiable"
        print(f"  {f['a']:>9s} - {f['b']:9s} {f['medida']:13s} {f['nota_a']:.4f} - {f['nota_b']:.4f} = "
              f"{f['diferencia']:+.4f}  corregido [{f['bajo_corregido']:+.4f}, {f['alto_corregido']:+.4f}] "
              f"{fiable}, gana a en el {f['gana_a']:.1%}")


def producto_por_partes():
    """Producto escalar construido con piezas ya entrenadas:
    coseno predicho por el modelo de la criba × norma predicha de a × norma predicha de b.
    Se compara con el modelo del producto escalar y con el del coseno, en medidas de orden.
    Bonferroni: 2 parejas × 2 medidas × 2 embeddings = 8 comparaciones."""
    test, media = test_con_objetivo()
    norma_predicha = np.load(CARPETA / f"predicciones/{EMBEDDING}_norma_RandomForest_B.npy")
    coseno = np.load(criba.CARPETA / f"predicciones/{EMBEDDING}_RandomForest_B.npy")
    predicciones = {
        "partes": coseno * norma_predicha[test["consulta"]] * norma_predicha[test["candidata"]],
        "producto": np.load(CARPETA / f"predicciones/{EMBEDDING}_producto_RandomForest_B.npy"),
        "coseno": coseno}
    notas = notas_de_modelos(test, predicciones, media)
    resultado = bootstrap_parejas(notas, [("partes", "producto"), ("partes", "coseno")],
                                  ["spearman", "precision_10"], comparaciones=8)
    resultado.to_csv(CARPETA / f"bootstrap_partes_{EMBEDDING}.csv", index=False)
    print(f"{EMBEDDING}: producto por partes, ordenando por el producto escalar real")
    mostrar(resultado)


def comparar_ampliado():
    """D38: modelo con 95 variables frente al de 35, con el mismo objetivo.
    Mismo objetivo, así que también vale el R².
    Bonferroni: 3 medidas × 2 objetivos × 2 embeddings = 12 comparaciones."""
    test, media = test_con_objetivo()
    base = (criba.CARPETA / f"predicciones/{EMBEDDING}_RandomForest_B.npy" if OBJETIVO == "coseno"
            else CARPETA / f"predicciones/{EMBEDDING}_{OBJETIVO}_RandomForest_B.npy")
    predicciones = {"ampliado": np.load(CARPETA / f"predicciones/{EMBEDDING}_{OBJETIVO}_ampliado_RandomForest_B.npy"),
                    "base": np.load(base)}
    notas = notas_de_modelos(test, predicciones, media)
    resultado = bootstrap_parejas(notas, [("ampliado", "base")], ["spearman", "precision_10", "r2"],
                                  comparaciones=12)
    resultado.to_csv(CARPETA / f"bootstrap_ampliado_{EMBEDDING}_{OBJETIVO}.csv", index=False)
    print(f"{EMBEDDING}, {OBJETIVO}: 95 variables (ampliado) frente a 35 (base)")
    mostrar(resultado)


AMPLIADOS = ["RandomForest_B", "LightGBM_C", "XGBoost_C"]


def comparar_modelos_ampliados():
    """D38: los tres principales con 95 variables. Se elige con la validación (regla D15,
    ya guardada en resultados.csv). El test solo describe: diferencias frente a RandomForest B.
    Bonferroni: 2 parejas × 3 medidas × 2 objetivos × 2 embeddings = 24 comparaciones."""
    test, media = test_con_objetivo()
    predicciones = {m.split("_")[0]: np.load(CARPETA / f"predicciones/{EMBEDDING}_{OBJETIVO}_ampliado_{m}.npy")
                    for m in AMPLIADOS}
    notas = notas_de_modelos(test, predicciones, media)
    resultado = bootstrap_parejas(notas, [("LightGBM", "RandomForest"), ("XGBoost", "RandomForest")],
                                  ["spearman", "precision_10", "r2"], comparaciones=24)
    resultado.to_csv(CARPETA / f"bootstrap_modelos_ampliados_{EMBEDDING}_{OBJETIVO}.csv", index=False)
    guardado = pd.read_csv(RESULTADOS)
    guardado = guardado[(guardado["embedding"] == EMBEDDING) & (guardado["objetivo"] == f"{OBJETIVO}_ampliado")]
    print(f"{EMBEDDING}, {OBJETIVO}, 95 variables: validación (decide) y test (describe)")
    print(guardado[["modelo", "segundos", "arboles", "r2_entrenamiento", "r2_validacion", "spearman_validacion",
                    "r2_test", "spearman_test", "precision_10_test"]].round(4).to_string(index=False))
    mostrar(resultado)


def curvas():
    """Curvas de aprendizaje de RandomForest B con el objetivo y las variables elegidas
    (35 o 95 con 'ampliado' delante: objetivos.py <emb> <obj> curvas [ampliado]).
    Mismo método que src/analisis.py: el 10, 25 y 50 % de las tablas de entrenamiento,
    3 semillas, se conservan los pares cuya columna consulta es de esas tablas, y se mide
    en la validación completa. El punto del 100 % sale de resultados.csv."""
    from analisis import FRACCIONES, SEMILLAS_CURVAS
    ampliado = len(sys.argv) > 4 and sys.argv[4] == "ampliado"
    nombre = f"{OBJETIVO}_ampliado" if ampliado else OBJETIVO
    archivo = CARPETA / f"curvas_{EMBEDDING}_{nombre}.csv"

    pares = anadir_objetivo(criba.obtener_pares())
    variables = [c for c in pares.columns if c not in criba.IDENTIFICADORES + ["objetivo"]]
    tabla_de = np.array([tabla for tabla, _ in pares.attrs["claves"]])
    entrenamiento = pares[pares["parte"] == "entrenamiento"]
    validacion = pares[pares["parte"] == "validacion"]

    def matriz(d):
        x = d[variables].to_numpy(dtype=np.float32)
        if ampliado:
            x = np.hstack([x, propios[d["consulta"].to_numpy()], propios[d["candidata"].to_numpy()]])
        return x

    propios = perfiles_numericos()[1] if ampliado else None
    x_val, y_val = matriz(validacion), validacion["objetivo"].to_numpy()
    media = entrenamiento["objetivo"].mean()
    tabla_consulta = tabla_de[entrenamiento["consulta"].to_numpy()]
    tablas = np.array(sorted(set(tabla_consulta)))

    filas = pd.read_csv(archivo).to_dict("records") if archivo.exists() else []
    hechas = {(f["fraccion"], f["semilla"]) for f in filas}
    for fraccion in FRACCIONES:
        for semilla in SEMILLAS_CURVAS:
            if (fraccion, semilla) in hechas:
                continue
            elegidas = np.random.default_rng(semilla).choice(tablas, round(fraccion * len(tablas)), replace=False)
            parte = entrenamiento[np.isin(tabla_consulta, elegidas)]
            x, y = matriz(parte), parte["objetivo"].to_numpy()
            inicio = time.time()
            modelo = criba.CONFIGURACIONES["RandomForest"]["B"]()
            modelo.fit(x, y)
            p_val = modelo.predict(x_val)
            filas.append({"fraccion": fraccion, "semilla": semilla, "tablas": len(elegidas), "pares": len(parte),
                          "segundos": round(time.time() - inicio, 1),
                          "r2_entrenamiento": criba.r2(y, modelo.predict(x), media),
                          "r2_validacion": criba.r2(y_val, p_val, media),
                          "spearman_validacion": criba.spearman_por_consulta(validacion, p_val, "objetivo")})
            pd.DataFrame(filas).to_csv(archivo, index=False)
            print(f"  {EMBEDDING} {nombre} {fraccion:.0%} semilla {semilla}: Spearman val "
                  f"{filas[-1]['spearman_validacion']:.3f} ({filas[-1]['segundos']:.0f} s)", flush=True)

    # Punto del 100 %: el entrenamiento completo ya hecho
    if OBJETIVO == "coseno" and not ampliado:
        completo = pd.read_csv(criba.RESULTADOS).query("modelo == 'RandomForest' and config == 'B'").iloc[0]
    else:
        completo = pd.read_csv(RESULTADOS).query(
            "embedding == @EMBEDDING and objetivo == @nombre and modelo == 'RandomForest_B'").iloc[0]
    curva = pd.DataFrame(filas).groupby("fraccion")[["pares", "spearman_validacion", "r2_validacion"]].mean()
    curva.loc[1.0] = [len(entrenamiento), completo["spearman_validacion"], completo["r2_validacion"]]
    print(f"{EMBEDDING}, {nombre}, {len(variables) + (60 if ampliado else 0)} variables:")
    print(curva.round(4).to_string())


if __name__ == "__main__":
    if OBJETIVO == "norma":
        norma()
    elif PASO == "comparar":
        comparar()
    elif PASO == "partes":
        producto_por_partes()
    elif PASO == "comparar_ampliado":
        comparar_ampliado()
    elif PASO == "comparar_modelos":
        comparar_modelos_ampliados()
    elif PASO == "curvas":
        curvas()
    else:
        main()
