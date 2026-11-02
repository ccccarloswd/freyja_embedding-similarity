"""Embeddings con MPNet, serializando cada columna como DeepJoin pero sin estadísticas.

Serialización: la de `4_embed_columns.py` de la implementación de DeepJoin del
repo de FREYJA (lectura de los CSV con latin1, todas las columnas de cada tabla),
quitando las estadísticas que DeepJoin escribe en el texto:

    DeepJoin: "<archivo>.csv information. <columna> contains <n> values (<máx>, <mín>, <media>): <v1>, <v2>, ..."
    Aquí:     "<archivo>.csv information. <columna>: <v1>, <v2>, ..."

  - v1, v2, ...: los valores distintos, en orden de aparición

Se quitan porque el número de valores distintos y el número de palabras por valor
también son métricas del perfil de FREYJA: si estuvieran escritos en el texto, el
modelo podría acertar el cosine copiando esa información (decisión D32).

Modelo: sentence-transformers/all-mpnet-base-v2, sin fine-tuning. Lee como mucho
384 tokens y corta el resto. Para no construir textos enormes con columnas de
millones de valores, el texto se recorta antes a MAX_CARACTERES: muy por encima
de lo que caben en 384 tokens, así que el embedding no cambia.

Salida en data/embeddings/<lake>_mpnet/: columnas.csv y original.npy (mismo
formato que los de WarpGate).

Uso (con el entorno .venv):  python src/embeddings_mpnet.py [freyjabm|opendata]

Guarda original.npy (normalizados, los del experimento) y sin_normalizar.npy
(para la distancia euclídea y el producto escalar). En OpenData solo se
calculan las columnas que ha perfilado FREYJA.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAKES = {"freyjabm": RAIZ / "data/raw/freyjabm/datalake",
         "opendata": RAIZ / "data/raw/lb_opendata/seleccion"}
LAKE = sys.argv[1] if len(sys.argv) > 1 else "freyjabm"
CARPETA_LAKE = LAKES[LAKE]
PERFILES = RAIZ / f"data/perfiles/{LAKE}/profiles_{LAKE}.csv"
SALIDA = RAIZ / "data/embeddings" / f"{LAKE}_mpnet"
MODELO = "sentence-transformers/all-mpnet-base-v2"
MAX_CARACTERES = 20000


def serializar(nombre_archivo, nombre_columna, columna):
    """Texto de DeepJoin para una columna, o None si no tiene valores."""
    celdas = columna.dropna().astype(str)
    if celdas.empty:
        return None
    valores = ", ".join(celdas.unique())[:MAX_CARACTERES]
    return f"{nombre_archivo} information. {nombre_columna}: {valores}."


def nombre_como_freyja(nombre):
    """El nombre de columna tal como lo guarda FREYJA (lee UTF-8 y limpia el nombre)."""
    try:
        nombre = str(nombre).encode("latin1").decode("utf-8")
    except UnicodeError:
        nombre = str(nombre)
    return (nombre.replace("\ufeff", "").replace('"', "").replace("\n", "")
            .replace("\r", "").strip())


def main():
    perfiladas = None
    if LAKE != "freyjabm":  # en OpenData solo las columnas que ha perfilado FREYJA
        perfiles = pd.read_csv(PERFILES, usecols=["dataset_name", "attribute_name"])
        perfiladas = set(zip(perfiles["dataset_name"], perfiles["attribute_name"]))

    textos, filas = [], []
    for archivo in sorted(a for a in os.listdir(CARPETA_LAKE) if a.endswith(".csv")):
        tabla = pd.read_csv(CARPETA_LAKE / archivo, encoding="latin1", low_memory=False)
        for posicion, nombre in enumerate(tabla.columns):
            nombre_freyja = nombre_como_freyja(nombre)
            if perfiladas is not None and (Path(archivo).stem, nombre_freyja) not in perfiladas:
                continue
            texto = serializar(archivo, nombre, tabla.iloc[:, posicion])
            if texto is None:
                continue
            textos.append(texto)
            filas.append({"tabla": archivo, "columna": nombre, "columna_freyja": nombre_freyja,
                          "posicion": posicion})

    modelo = SentenceTransformer(MODELO, device="cpu")
    tokens = [len(modelo.tokenizer(t, truncation=False)["input_ids"]) for t in textos]
    # all-mpnet-base-v2 termina con un paso que normaliza los vectores (longitud 1).
    # Se calculan sin ese paso y se normalizan aparte: así se guardan las dos versiones.
    sin_normalizar = SentenceTransformer(modules=list(modelo)[:-1], device="cpu").encode(
        textos, batch_size=32, show_progress_bar=False)
    normalizados = sin_normalizar / np.linalg.norm(sin_normalizar, axis=1, keepdims=True)

    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA / "columnas.csv", index=False)
    np.save(SALIDA / "original.npy", normalizados)
    np.save(SALIDA / "sin_normalizar.npy", sin_normalizar)
    embeddings = normalizados
    if perfiladas is not None:
        print(f"Columnas perfiladas por FREYJA: {len(perfiladas)}; con embedding: {len(filas)}")

    tokens = np.array(tokens)
    print(f"Columnas: {len(textos)}; embeddings: {embeddings.shape}")
    print(f"Tokens por texto: mediana {np.median(tokens):.0f}; "
          f"cortados por superar {modelo.max_seq_length}: {(tokens > modelo.max_seq_length).mean():.0%}")
    print("Ejemplo:", textos[0][:300])
    print(f"Escrito en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
