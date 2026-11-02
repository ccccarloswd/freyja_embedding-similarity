"""Gráficos del README (etiquetas en inglés) a partir de los resultados guardados.

  - Fases 1 y 2 (FREYJABM, validación cruzada): real frente a predicho, error
    por zonas de cosine e importancia de las variables. Usa las predicciones de
    experimento.py (v1, WarpGate y MPNet).
  - Criba de modelos (OpenData): Spearman por consulta en validación de la mejor
    configuración de cada modelo. Usa data/criba/resultados_<embedding>.csv.

Uso (entorno .venv-ml):  python src/figuras.py
Salida: docs/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.argv = sys.argv[:1]  # experimento.py lee la variante de los argumentos
sys.path.insert(0, str(Path(__file__).resolve().parent))
import informe  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
SALIDA = RAIZ / "docs/figures"

NOMBRES = {"warpgate": "WarpGate (TaBERT-K3)", "mpnet": "MPNet (all-mpnet-base-v2)"}
COLORES = {"warpgate": "#2a78d6", "mpnet": "#eb6834"}
REFERENCIA = "#8a8984"
SUPERFICIE, TEXTO, TEXTO_2, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#eef4fc", "#7fb0ec", "#2a78d6", "#0f3d75"])

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "text.color": TEXTO,
    "xtick.color": TEXTO_2, "ytick.color": TEXTO_2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

MODELOS_CRIBA = {
    "GradientBoosting (FREYJA, referencia)": "GradientBoosting (FREYJA settings)",
    "RandomForest": "RandomForest", "ExtraTrees": "ExtraTrees",
    "HistGradientBoosting": "HistGradientBoosting", "XGBoost": "XGBoost", "LightGBM": "LightGBM",
    "RealMLP": "RealMLP", "KNeighbors (prueba rápida)": "KNeighbors",
}


def guardar(figura, nombre):
    figura.savefig(SALIDA / nombre, dpi=160, bbox_inches="tight")
    plt.close(figura)
    print("Escrito", SALIDA / nombre)


def fases_1_y_2():
    pares = {}
    for modelo in NOMBRES:
        datos, _ = informe.pares_con_predicciones(modelo)
        pares[modelo] = datos.dropna(subset=["prediccion"])

    # Real frente a predicho
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.4))
    for eje, (modelo, datos) in zip(ejes, pares.items()):
        bajo, alto = np.percentile(datos["coseno"], [0.5, 99.5])
        eje.hexbin(datos["coseno"], datos["prediccion"], gridsize=60, cmap=SECUENCIAL, bins="log",
                   mincnt=1, extent=(bajo, alto, bajo, alto), linewidths=0)
        eje.plot([bajo, alto], [bajo, alto], color=TEXTO_2, linewidth=1, linestyle="--")
        eje.set(xlim=(bajo, alto), ylim=(bajo, alto), xlabel="Real cosine similarity",
                ylabel="Predicted from profiles", title=NOMBRES[modelo])
        eje.grid(False)
        eje.text(0.03, 0.95, "dashed line: perfect prediction", transform=eje.transAxes,
                 color=TEXTO_2, fontsize=8.5, va="top")
    figura.suptitle("Phases 1-2 on FREYJABM: predictions regress towards the mean", x=0.01,
                    ha="left", fontweight="bold", fontsize=12)
    figura.text(0.01, -0.02, "Held-out pairs of a 5-fold cross-validation grouped by table. "
                "Colour: number of pairs (log scale).", color=TEXTO_2, fontsize=8.5)
    guardar(figura, "phase1_real_vs_predicted.png")

    # Error con signo por deciles del cosine real
    figura, eje = plt.subplots(figsize=(7.5, 4))
    for modelo, datos in pares.items():
        decil = pd.qcut(datos["coseno"].rank(method="first"), 10, labels=False) + 1
        error = (datos["prediccion"] - datos["coseno"]).groupby(decil).mean()
        eje.plot(error.index, error.values, color=COLORES[modelo], linewidth=2, marker="o",
                 markersize=6, markeredgecolor=SUPERFICIE, markeredgewidth=1.5, label=NOMBRES[modelo])
        eje.annotate(f"{error.iloc[-1]:+.2f}", (10, error.iloc[-1]), xytext=(8, 0),
                     textcoords="offset points", va="center", color=TEXTO_2, fontsize=9)
    eje.axhline(0, color=TEXTO_2, linewidth=1)
    eje.set(xticks=range(1, 11), xlabel="Decile of the real cosine (1 = least similar pairs)",
            ylabel="Mean signed error (predicted − real)",
            title="Error by similarity decile: over-estimates dissimilar pairs, under-estimates similar ones")
    eje.legend(frameon=False, loc="lower left")
    guardar(figura, "phase1_error_by_decile.png")

    # Importancia de las variables (permutación, media de las 5 rondas)
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.2))
    for eje, modelo in zip(ejes, NOMBRES):
        importancias = pd.read_csv(DATOS / f"experimento/importancias_freyjabm_{modelo}_v1.csv")
        media = importancias.groupby("variable")["importancia_permutacion"].mean().nlargest(10)[::-1]
        eje.barh(media.index, media.values, color=COLORES[modelo], height=0.6)
        eje.set(xlabel="Drop in R² when the feature is shuffled", title=NOMBRES[modelo])
        eje.grid(axis="y", visible=False)
        eje.tick_params(axis="y", length=0)
    figura.suptitle("Phases 1-2: each embedding relies on the profile features that describe what it reads",
                    x=0.01, ha="left", fontweight="bold", fontsize=12)
    figura.tight_layout()
    guardar(figura, "phase1_feature_importance.png")


def criba():
    mejores = {}
    for modelo in NOMBRES:
        resultados = pd.read_csv(DATOS / f"criba/resultados_{modelo}.csv")
        resultados = resultados[resultados["modelo"].isin(MODELOS_CRIBA)]
        mejores[modelo] = (resultados.sort_values("spearman_validacion")
                           .groupby("modelo").tail(1).set_index("modelo"))
    orden = mejores["warpgate"]["spearman_validacion"].sort_values().index

    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for eje, (modelo, tabla) in zip(ejes, mejores.items()):
        referencia = tabla.loc["GradientBoosting (FREYJA, referencia)", "spearman_validacion"]
        eje.axvline(referencia, color=REFERENCIA, linewidth=1, linestyle="--")
        for fila, nombre in enumerate(orden):
            if nombre not in tabla.index:
                eje.text(0.355, fila, "not run", va="center", color=TEXTO_2, fontsize=8.5)
                continue
            valor = tabla.loc[nombre, "spearman_validacion"]
            color = REFERENCIA if "referencia" in nombre else COLORES[modelo]
            eje.plot([0.35, valor], [fila, fila], color=REJILLA, linewidth=2, zorder=1)
            eje.scatter(valor, fila, s=70, color=color, edgecolor=SUPERFICIE, linewidth=1.5, zorder=2)
            eje.text(valor + 0.006, fila, f"{valor:.3f} ({tabla.loc[nombre, 'config']})",
                     va="center", color=TEXTO_2, fontsize=8.5)
        eje.set(xlim=(0.35, 0.62), title=NOMBRES[modelo], xlabel="Spearman per query column (validation)")
        eje.grid(axis="y", visible=False)
        eje.tick_params(axis="y", length=0)
    ejes[0].set_yticks(range(len(orden)), [MODELOS_CRIBA[n] for n in orden])
    figura.suptitle("Model screening on LakeBench OpenData: different models hit the same ceiling",
                    x=0.01, ha="left", fontweight="bold", fontsize=12)
    figura.text(0.01, -0.03, "Best of 3 configurations per model (in brackets), chosen on validation. "
                "Dashed line: GradientBoosting with FREYJA's settings.", color=TEXTO_2, fontsize=8.5)
    figura.tight_layout()
    guardar(figura, "screening_spearman.png")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    criba()
    fases_1_y_2()


if __name__ == "__main__":
    main()
