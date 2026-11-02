"""Repositorio de GitHub del que procede cada tabla de GitTables.

La versión CSV de GitTables no indica el origen de cada tabla. La versión
Parquet (Zenodo 6517052) sí: cada archivo guarda en sus metadatos la URL del
CSV original en GitHub. Los nombres de archivo coinciden entre ambas versiones.

Este script descarga cada tema de la versión Parquet, lee solo esos metadatos
y borra la descarga. Si se interrumpe, al volver a ejecutarlo continúa con
los temas que falten. Escribe data/gittables_repos.csv con una fila por tabla:
tema, archivo (nombre del CSV), repositorio (usuario/repo), licencia y URL.

Uso:  python src/repos_gittables.py
"""
import json
import re
import struct
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "gittables_repos.csv"
REGISTRO = "https://zenodo.org/api/records/6517052"

URL_CSV = re.compile(rb'"csv_url": ?"([^"]+)"')
LICENCIA = re.compile(rb'"license": ?"([^"]*)"')
REPOSITORIO = re.compile(r"github(?:usercontent)?\.com/([^/]+/[^/]+)/")


def metadatos_parquet(contenido):
    """Pie de un archivo Parquet: ahí están los metadatos, sin comprimir."""
    longitud = struct.unpack("<I", contenido[-8:-4])[0]
    return contenido[-8 - longitud:-8]


def procesar_tema(archivo):
    tema = archivo["key"].removesuffix("_tables_licensed.zip")
    filas = []
    with tempfile.TemporaryDirectory() as carpeta:
        destino = Path(carpeta) / "tema.zip"
        for intento in range(8):
            try:
                urllib.request.urlretrieve(archivo["links"]["self"], destino)
                if destino.stat().st_size == archivo["size"]:
                    break
            except OSError:
                pass
            time.sleep(60)  # Zenodo limita el número de descargas por minuto
        else:
            return tema, None
        with zipfile.ZipFile(destino) as comprimido:
            for entrada in comprimido.infolist():
                if not entrada.filename.endswith(".parquet"):
                    continue
                pie = metadatos_parquet(comprimido.read(entrada))
                url = URL_CSV.search(pie)
                licencia = LICENCIA.search(pie)
                url = url.group(1).decode("utf-8", "replace") if url else ""
                repositorio = REPOSITORIO.search(url)
                filas.append({
                    "tema": tema,
                    "archivo": Path(entrada.filename).stem + ".csv",
                    "repositorio": repositorio.group(1).lower() if repositorio else "",
                    "licencia": licencia.group(1).decode("utf-8", "replace") if licencia else "",
                    "url": url,
                })
    return tema, filas


def main():
    with urllib.request.urlopen(REGISTRO) as respuesta:
        archivos = json.load(respuesta)["files"]
    # Si ya existe una salida parcial, solo se procesan los temas que faltan
    previas = pd.read_csv(SALIDA, keep_default_na=False) if SALIDA.exists() else None
    if previas is not None:
        hechos = set(previas["tema"])
        archivos = [a for a in archivos
                    if a["key"].removesuffix("_tables_licensed.zip") not in hechos]
    filas, fallidos = [], []
    with ThreadPoolExecutor(max_workers=2) as ejecutor:
        for tema, filas_tema in ejecutor.map(procesar_tema, archivos):
            if filas_tema is None:
                fallidos.append(tema)
                print(f"FALLO {tema}", flush=True)
            else:
                filas.extend(filas_tema)
                print(f"{tema}: {len(filas_tema)} tablas", flush=True)
    tabla = pd.concat([previas, pd.DataFrame(filas)]) if previas is not None else pd.DataFrame(filas)
    tabla.to_csv(SALIDA, index=False)
    print(f"\nTablas: {len(tabla)}; sin repositorio: {(tabla['repositorio'] == '').sum()}; "
          f"repositorios distintos: {tabla['repositorio'].nunique()}")
    print(f"Temas fallidos: {fallidos}")
    print(f"Escrito en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
