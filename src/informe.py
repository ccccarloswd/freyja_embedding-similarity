"""Resumen de resultados de las fases 1 y 2 para compartir (tablas y gráficos).

Lee las predicciones guardadas por experimento.py (v1, WarpGate y MPNet) y
calcula, además de MAE, R² y Spearman por consulta:
  - Precision@10: para cada columna consulta, qué parte de sus 10 candidatas
    más parecidas según el embedding están también entre las 10 primeras según
    el modelo. Se compara con lo que daría un orden al azar.
  - Comportamiento en la zona alta: en el 10 % de pares con cosine más alto,
    error medio con signo (negativo = el modelo se queda corto).

Gráficos en data/informe/: real frente a predicho, e importancia de variables.

Uso (con el entorno freyja):  python src/informe.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.argv = sys.argv[:1]  # experimento.py lee la variante de los argumentos
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experimento as e  # noqa: E402

SALIDA = e.DATOS / "informe"
MODELOS = {"warpgate": "WarpGate (TaBERT)", "mpnet": "MPNet"}
COLORES = {"warpgate": "#2a78d6", "mpnet": "#eb6834"}  # paleta de referencia, huecos 1 y 2
SUPERFICIE, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
K = 10


def pares_con_predicciones(modelo):
    perfiles = pd.read_pickle(e.PERFILES)
    embeddings_todos = np.load(e.DATOS / f"embeddings/freyjabm_{modelo}/original.npy")
    columnas = pd.read_csv(e.DATOS / f"embeddings/freyjabm_{modelo}/columnas.csv")
    fila = {(t.removesuffix(".csv"), c): i for i, (t, c) in
            enumerate(zip(columnas["tabla"], columnas["columna"]))}
    embeddings = np.array([embeddings_todos[fila[(d, a)]] for d, a in
                           zip(perfiles["dataset_name"], perfiles["attribute_name"])])
    pares = e.obtener_distancias(perfiles)
    pares["coseno"] = e.calcular_cosenos(pares, embeddings)
    grupo_tabla = e.repartir_tablas(perfiles)
    grupo_columna = [grupo_tabla[t] for t in perfiles["dataset_name"]]
    pares["grupo_consulta"] = [grupo_columna[i] for i in pares["consulta"]]
    pares["grupo_candidata"] = [grupo_columna[i] for i in pares["candidata"]]
    pares["prediccion"] = pd.read_csv(
        e.DATOS / f"experimento/predicciones_freyjabm_{modelo}_v1.csv")["prediccion"]
    return pares, perfiles


def precision_en_k(examen):
    aciertos, azar = [], []
    for _, sus_pares in examen.groupby("consulta"):
        reales = set(sus_pares.nlargest(K, "coseno")["candidata"])
        predichas = set(sus_pares.nlargest(K, "prediccion")["candidata"])
        aciertos.append(len(reales & predichas) / K)
        azar.append(K / len(sus_pares))
    return np.mean(aciertos), np.mean(azar)


def resumen(modelo):
    pares, perfiles = pares_con_predicciones(modelo)
    filas = []
    for grupo in range(e.GRUPOS):
        entrenamiento, examen = e.separar_ronda(pares, grupo)
        media = pares.loc[entrenamiento, "coseno"].mean()
        notas = e.evaluar_ronda(pares[examen], media)
        notas["precision_10"], notas["precision_10_azar"] = precision_en_k(pares[examen])
        alto = pares[examen]
        alto = alto[alto["coseno"] >= alto["coseno"].quantile(0.9)]
        notas["sesgo_zona_alta"] = (alto["prediccion"] - alto["coseno"]).mean()
        filas.append(notas)
    return pd.DataFrame(filas), pares, perfiles


def estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.grid(color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)


def grafico_real_frente_a_predicho(todos):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SUPERFICIE)
    for ax, (modelo, pares) in zip(ejes, todos.items()):
        con = pares.dropna(subset=["prediccion"])
        estilo(ax)
        ax.hexbin(con["coseno"], con["prediccion"], gridsize=45, mincnt=1, bins="log",
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                      "", ["#f3f2ef", COLORES[modelo]]), linewidths=0)
        limite = [min(con["coseno"].min(), con["prediccion"].min()),
                  max(con["coseno"].max(), con["prediccion"].max())]
        ax.plot(limite, limite, color=TEXTO_2, linewidth=1, linestyle="--")
        ax.text(limite[1], limite[1], " real = predit", color=TEXTO_2, fontsize=8,
                ha="right", va="bottom")
        ax.set_title(MODELOS[modelo], color=TEXTO, fontsize=11, loc="left")
        ax.set_xlabel("Cosine real", color=TEXTO_2, fontsize=9)
        ax.set_ylabel("Cosine predit", color=TEXTO_2, fontsize=9)
    fig.suptitle("Cosine real i predit (parelles de test de les 5 rondes)",
                 color=TEXTO, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SALIDA / "real_i_predit.png", dpi=160, facecolor=SUPERFICIE)


def grafico_importancias():
    medias = {}
    for modelo in MODELOS:
        imp = pd.read_csv(e.DATOS / f"experimento/importancias_freyjabm_{modelo}_v1.csv")
        medias[modelo] = imp.groupby("variable")["importancia_permutacion"].mean()
    tabla = pd.DataFrame(medias)
    principales = tabla.max(axis=1).sort_values(ascending=False).head(10).index[::-1]
    tabla = tabla.loc[principales]

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SUPERFICIE)
    estilo(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(tabla))
    alto = 0.36
    for desplazamiento, modelo in ((alto / 2 + 0.02, "warpgate"), (-alto / 2 - 0.02, "mpnet")):
        ax.barh(y + desplazamiento, tabla[modelo].clip(lower=0), height=alto,
                color=COLORES[modelo], label=MODELOS[modelo])
    ax.set_yticks(y, tabla.index, color=TEXTO, fontsize=9)
    ax.set_xlabel("R² que es perd en desordenar la variable (mitjana de les 5 rondes)",
                  color=TEXTO_2, fontsize=9)
    ax.set_title("Quines variables del perfil fa servir el model amb cada embedding",
                 color=TEXTO, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO, loc="lower right")
    fig.tight_layout()
    fig.savefig(SALIDA / "importancia_variables.png", dpi=160, facecolor=SUPERFICIE)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    todos, tablas = {}, []
    for modelo in MODELOS:
        notas, pares, perfiles = resumen(modelo)
        todos[modelo] = pares
        media, desviacion = notas.mean(), notas.std()
        fila = {"embedding": MODELOS[modelo]}
        for nota in notas.columns:
            fila[nota] = f"{media[nota]:.3f} ({desviacion[nota]:.3f})"
        tablas.append(fila)
        print(f"\n{MODELOS[modelo]}, por ronda:")
        print(notas.round(3).to_string())
        print(f"Cosine: media {pares['coseno'].mean():.3f}, desv. {pares['coseno'].std():.3f}")
    tabla = pd.DataFrame(tablas).set_index("embedding")
    tabla.to_csv(SALIDA / "resumen.csv")
    print("\nMedia (desviación típica) de las 5 rondas:")
    print(tabla.T.to_string())

    grafico_real_frente_a_predicho(todos)
    grafico_importancias()
    print(f"\nEscrito en {SALIDA}")


if __name__ == "__main__":
    main()
