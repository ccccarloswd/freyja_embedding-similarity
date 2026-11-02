"""Embeddings de WarpGate (TaBERT-K3) para las columnas de un data lake.

Reproduce el cálculo de `1_create_embeddings.py` del repo de FREYJA: misma
lectura de los CSV (latin1, tipos adivinados por pandas), misma limpieza de
valores y misma llamada a TaBERT, con todas las columnas de cada tabla.

Única diferencia: a TaBERT solo se le pasan las filas que va a usar. El modelo
K3 lee las tres primeras e ignora el resto (comprobado: el embedding sale
idéntico), y así se evita trocear en tokens columnas de un millón de filas.
Al empezar se verifica esa equivalencia con unas cuantas columnas.

Además de la variante original se calculan tres variantes de diagnóstico:
  - original:     nombre real, tres primeras filas (lo que hace FREYJA)
  - filas_azar_1: nombre real, tres filas elegidas al azar
  - filas_azar_2: nombre real, otras tres filas elegidas al azar
  - nombre_fijo:  tres primeras filas, con el mismo nombre para todas las columnas

Salida en data/embeddings/<lake>_warpgate/: columnas.csv y un .npy por variante,
con las filas en el mismo orden.

Necesita la carpeta warpgate/ (código de TaBERT) y el modelo en data/modelos/.
Uso (con el entorno .venv):  python src/embeddings_warpgate.py [freyjabm|opendata]

Con opendata solo se calcula la variante original, y solo para las columnas que
ha perfilado FREYJA.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parent.parent
# Código de TaBERT de la implementación de WarpGate del repo de FREYJA (no incluido):
# variable de entorno WARPGATE_DIR, o por defecto la carpeta warpgate/ de este proyecto
WARPGATE = Path(os.environ.get("WARPGATE_DIR", RAIZ / "warpgate"))
sys.path.insert(0, str(WARPGATE))
sys.path.insert(0, str(WARPGATE / "TaBERT"))
from TaBERT.table_bert import Column, Table  # noqa: E402
from TaBERT.table_bert.table_bert import TableBertModel  # noqa: E402

# Lake a procesar (primer argumento): carpeta, variantes y si se limita a las
# columnas que ha perfilado FREYJA (las demás no se usan en el experimento)
LAKES = {
    "freyjabm": dict(carpeta=RAIZ / "data/raw/freyjabm/datalake", solo_perfiladas=False,
                     variantes=["original", "filas_azar_1", "filas_azar_2", "nombre_fijo"]),
    "opendata": dict(carpeta=RAIZ / "data/raw/lb_opendata/seleccion", solo_perfiladas=True,
                     variantes=["original"]),
}
LAKE = sys.argv[1] if len(sys.argv) > 1 else "freyjabm"
CARPETA_LAKE = LAKES[LAKE]["carpeta"]
VARIANTES = LAKES[LAKE]["variantes"]
PERFILES = RAIZ / f"data/perfiles/{LAKE}/profiles_{LAKE}.csv"
MODELO = RAIZ / "data/modelos/tabert_large_k3/model.bin"
SALIDA = RAIZ / "data/embeddings" / f"{LAKE}_warpgate"

SEMILLA = 0
NOMBRE_FIJO = "column"

model = TableBertModel.from_pretrained(str(MODELO))
model.eval()
FILAS = model.config.sample_row_num  # 3 en TaBERT-K3


def nombre_como_freyja(nombre):
    """El nombre de columna tal como lo guarda FREYJA en sus perfiles.

    El CSV se lee con latin1 (como WarpGate), pero FREYJA lo lee como UTF-8 y
    además limpia el nombre (comillas, saltos de línea, espacios).
    """
    try:
        nombre = str(nombre).encode("latin1").decode("utf-8")
    except UnicodeError:
        nombre = str(nombre)
    return (nombre.replace("\ufeff", "").replace('"', "").replace("\n", "")
            .replace("\r", "").strip())


def columnas_perfiladas():
    perfiles = pd.read_csv(PERFILES, usecols=["dataset_name", "attribute_name"])
    return set(zip(perfiles["dataset_name"], perfiles["attribute_name"]))


def valores_no_nulos(columna):
    """Misma limpieza que FREYJA: sin nulos, como texto y sin cadenas vacías."""
    valores = columna.dropna().astype(str)
    return valores[valores.str.strip() != ""]


def embedding(nombre, valores):
    """Embedding de una columna, igual que `column_embedding` de FREYJA."""
    tabla = Table(
        id="table",
        header=[Column(nombre, "text", sample_value=valores[0])],
        data=[[v] for v in valores],
    ).tokenize(model.tokenizer)
    with torch.no_grad():
        _, columnas, _ = model.encode(contexts=[model.tokenizer.tokenize("")], tables=[tabla])
    return columnas[0][0, :].cpu().numpy()


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas, vectores = [], {variante: [] for variante in VARIANTES}
    comprobadas = 0
    perfiladas = columnas_perfiladas() if LAKES[LAKE]["solo_perfiladas"] else None
    archivos = sorted(a for a in os.listdir(CARPETA_LAKE) if a.endswith(".csv"))
    for numero, archivo in enumerate(archivos, 1):
        tabla = pd.read_csv(CARPETA_LAKE / archivo, encoding="latin1", low_memory=False)
        for posicion, nombre in enumerate(tabla.columns):
            nombre_freyja = nombre_como_freyja(nombre)
            if perfiladas is not None and (Path(archivo).stem, nombre_freyja) not in perfiladas:
                continue
            valores = valores_no_nulos(tabla.iloc[:, posicion])
            if valores.empty:
                continue
            valores = valores.tolist()
            indice = len(filas)
            primeras = valores[:FILAS]
            original = embedding(nombre, primeras)

            if comprobadas < 5 and FILAS < len(valores) <= 5000:
                # Equivalencia con pasar la columna entera, como hace FREYJA
                assert np.array_equal(original, embedding(nombre, valores)), (archivo, nombre)
                comprobadas += 1

            vectores["original"].append(original)
            filas.append({"tabla": archivo, "columna": nombre, "columna_freyja": nombre_freyja,
                          "posicion": posicion, "no_nulos": len(valores)})
            if "filas_azar_1" not in VARIANTES:
                continue
            for intento in (1, 2):
                azar = np.random.default_rng([SEMILLA, indice, intento])
                elegidas = np.sort(azar.choice(len(valores), min(FILAS, len(valores)),
                                               replace=False))
                vectores[f"filas_azar_{intento}"].append(
                    embedding(nombre, [valores[i] for i in elegidas]))
            vectores["nombre_fijo"].append(embedding(NOMBRE_FIJO, primeras))
        print(f"{numero}/{len(archivos)} {archivo}: {len(filas)} columnas", flush=True)

    pd.DataFrame(filas).to_csv(SALIDA / "columnas.csv", index=False)
    for variante, lista in vectores.items():
        np.save(SALIDA / f"{variante}.npy", np.stack(lista))
    print(f"Equivalencia comprobada en {comprobadas} columnas")
    if perfiladas is not None:
        print(f"Columnas perfiladas por FREYJA: {len(perfiladas)}; con embedding: {len(filas)}")
    print(f"Escrito en {SALIDA}: {len(filas)} columnas, {len(VARIANTES)} variantes")


if __name__ == "__main__":
    sys.exit(main())
