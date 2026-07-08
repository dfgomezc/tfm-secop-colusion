#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye el modelo dimensional a partir de la capa bronze.

Arquitectura de la salida (deliberada):
  * Los datos viven en OUTPUTS/gold/<tabla>/*.parquet  -> los lee Power BI.
  * gold/secop_tfm.duckdb es una base LIGERA que solo contiene VISTAS sobre
    esos parquet. Asi la base pesa unos KB, se regenera al instante y ambos
    consumidores (DuckDB y Power BI) ven exactamente los mismos datos.

Pasos:
  1. Vistas staging (stg_*) sobre bronze/*.parquet con dedup global por clave.
  2. Ejecuta 02_gold.sql: cada CREATE TABLE se materializa a parquet y se
     re-expone como vista; cada CREATE VIEW se define sobre esas vistas.
  3. Escribe el .duckdb con todas las definiciones.

Uso:
  python -m tfm.ingesta.gold                                  # todo de una vez
  python -m tfm.ingesta.gold --ram 12GB --hilos 8
  python -m tfm.ingesta.gold --trozos 4                       # menos memoria por pasada
  python -m tfm.ingesta.gold --reanudar --limite-seg 300      # por tramos
  python -m tfm.ingesta.gold --materializar                   # tambien las vistas v_*
"""

import argparse
import glob
import os
import re
import tempfile
import time

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
#: La capa bronze es la entrada y el modelo dimensional la salida. Los dos
#: viven bajo `OUTPUTS/`, en temas distintos.
RAIZ_DEF = rutas.bronze()
GOLD_DEF = rutas.parquet()

# Clave de negocio para el DISTINCT global de cada staging.
STAGING = {
    "stg_procesos":    ("procesos",    ["id_del_proceso"]),
    "stg_contratos":   ("contratos",   ["id_contrato"]),
    "stg_proponentes": ("proponentes", ["id_procedimiento", "codigo_proveedor"]),
    "stg_ofertas":     ("ofertas",     ["id_del_proceso", "id_oferta", "codigo_proveedor"]),
    "stg_grupos":      ("grupos",      ["codigo_grupo", "codigo_participante"]),
    "stg_proveedores": ("proveedores", ["codigo_proveedor"]),
    "stg_contacto":    ("contacto",    ["codigo_entidad"]),
    "stg_multas":      ("multas",      None),
}


def parquet_valido(ruta):
    try:
        if os.path.getsize(ruta) < 8:
            return False
        with open(ruta, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def parquets(carpeta):
    """Parquet completos de una carpeta; descarta vacios y truncados."""
    todos = sorted(glob.glob(os.path.join(carpeta, "*.parquet")))
    buenos = [f for f in todos if parquet_valido(f)]
    for f in todos:
        if f not in buenos:
            print(f"  [!] parquet incompleto, se ignora: {os.path.basename(f)}")
    return buenos


def lista_sql(fs):
    return "[" + ", ".join("'" + f.replace("\\", "/").replace("'", "''") + "'"
                           for f in fs) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ_DEF)
    ap.add_argument("--bronze", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--trozos", type=int, default=1,
                    help="escribe cada tabla en N pasadas (menos memoria)")
    ap.add_argument("--materializar", action="store_true",
                    help="tambien vuelca las vistas analiticas v_* a parquet")
    ap.add_argument("--reanudar", action="store_true",
                    help="conserva las tablas ya volcadas a parquet")
    ap.add_argument("--limite-seg", type=int, default=0, dest="limite",
                    help="presupuesto global en segundos (usar con --reanudar)")
    a = ap.parse_args()

    bronze = a.bronze or a.raiz
    pq = a.gold or GOLD_DEF
    os.makedirs(pq, exist_ok=True)

    tmp = os.path.join(tempfile.gettempdir(), "duckdb_secop")
    os.makedirs(tmp, exist_ok=True)

    con = duckdb.connect()                      # en memoria: solo orquesta
    con.execute(f"SET memory_limit='{a.ram}'")
    con.execute("SET preserve_insertion_order=false")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")
    con.execute(f"SET temp_directory='{tmp}'")

    print(f"GOLD  bronze={bronze}\n      destino={pq}\n")

    # ---- 1. staging -------------------------------------------------------
    print("[staging]")
    for vista, (carpeta, claves) in STAGING.items():
        fs = parquets(os.path.join(bronze, carpeta))
        if not fs:
            print(f"  [!] sin parquet en bronze/{carpeta} -> se omite")
            continue
        fuente = f"read_parquet({lista_sql(fs)}, union_by_name=true)"
        if not claves:
            q = f"SELECT * FROM {fuente}"
        else:
            # Un QUALIFY row_number() sobre todo el dataset obliga a ordenar
            # TODAS las columnas y desborda el disco temporal. Como los
            # duplicados son residuales (paginacion de Socrata), se localizan
            # primero las claves repetidas -leyendo solo esas columnas- y la
            # ventana se aplica unicamente a ese subconjunto minusculo.
            k = ", ".join(claves)
            cond = " AND ".join(
                f"s.{c} IS NOT DISTINCT FROM d.{c}" for c in claves)
            q = f"""
WITH _dup AS (
    SELECT {k} FROM {fuente} GROUP BY {k} HAVING count(*) > 1
)
SELECT * FROM {fuente} s
WHERE NOT EXISTS (SELECT 1 FROM _dup d WHERE {cond})
UNION ALL BY NAME
SELECT * FROM (
    SELECT * FROM {fuente} s
    WHERE EXISTS (SELECT 1 FROM _dup d WHERE {cond})
    QUALIFY row_number() OVER (PARTITION BY {k}) = 1
)"""
        con.execute(f"CREATE OR REPLACE VIEW {vista} AS {q}")
        # conteo por metadatos del parquet: no escanea los datos
        n = con.execute("SELECT sum(num_rows) FROM parquet_file_metadata("
                        f"{lista_sql(fs)})").fetchone()[0]
        print(f"  {vista:<18} {n:>12,}  ({len(fs)} ficheros, antes de dedup)")

    # ---- 2. modelo --------------------------------------------------------
    print("\n[modelo]")
    with open(os.path.join(AQUI, "gold.sql"), "r", encoding="utf-8") as fh:
        sentencias = [s for s in fh.read().split(";\n")
                      if re.search(r"\bCREATE\b", s, re.I)]

    fin = (time.time() + a.limite) if a.limite else 0
    pendientes, definiciones = 0, []

    def exponer(nombre):
        """Publica como vista el parquet ya volcado de una tabla."""
        fs = parquets(os.path.join(pq, nombre))
        sql = f"SELECT * FROM read_parquet({lista_sql(fs)}, union_by_name=true)"
        con.execute(f"CREATE OR REPLACE VIEW {nombre} AS {sql}")
        definiciones.append((nombre, nombre))

    for s in sentencias:
        m = re.search(r"(TABLE|VIEW|MACRO)\s+(?:IF NOT EXISTS\s+)?([A-Za-z_0-9]+)",
                      s, re.I)
        objeto = m.group(2) if m else "?"
        clase = m.group(1).upper() if m else ""

        # --- macros: se recrean siempre (son baratas) ---
        if clase == "MACRO":
            con.execute(s + ";")
            definiciones.append(("MACRO", s.strip()))
            continue

        # --- vistas analiticas: dependen de las tablas ---
        if clase == "VIEW":
            if pendientes:
                continue
            con.execute(s + ";")
            definiciones.append(("VIEW", s.strip()))
            if a.materializar:
                volcar(con, objeto, f"SELECT * FROM {objeto}", pq, 1)
            continue

        # --- tablas: se materializan a parquet ---
        carpeta = os.path.join(pq, objeto)
        marca = os.path.join(carpeta, "_OK")
        if a.reanudar and os.path.exists(marca):
            exponer(objeto)
            n = con.execute(f"SELECT count(*) FROM {objeto}").fetchone()[0]
            print(f"  {objeto:<28} {n:>12,}  (ya existe)")
            continue
        if fin and time.time() > fin:
            pendientes += 1
            continue

        t0 = time.time()
        cuerpo = re.split(r"\bAS\b", s, maxsplit=1, flags=re.I)[1]
        if not volcar(con, objeto, cuerpo, pq, a.trozos, fin, a.reanudar):
            pendientes += 1
            continue
        exponer(objeto)
        n = con.execute(f"SELECT count(*) FROM {objeto}").fetchone()[0]
        print(f"  {objeto:<28} {n:>12,}  ({time.time()-t0:.1f}s)", flush=True)

    if pendientes:
        print(f"\n[!] quedan {pendientes} tablas; relanza con --reanudar")
        return

    # ---- 3. base DuckDB ligera con las vistas -----------------------------
    #: La base va junto a los parquet a los que apunta. Es una comodidad
    #: local: sus vistas guardan rutas absolutas y no resuelven en otra
    #: máquina, por eso no viaja en el paquete de datos.
    bd = os.path.join(pq, "secop_tfm.duckdb")
    for resto in (bd, bd + ".wal"):
        if os.path.exists(resto):
            try:
                os.remove(resto)
            except OSError:
                pass
    salida = duckdb.connect(bd)
    for etiqueta, sql in definiciones:
        if etiqueta in ("MACRO", "VIEW"):
            salida.execute(sql + ";")
        else:
            fs = parquets(os.path.join(pq, sql))
            salida.execute(
                f"CREATE OR REPLACE VIEW {sql} AS SELECT * FROM "
                f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    salida.close()

    print(f"\nOK\n  datos : {pq}\n  base  : {bd}")


def volcar(con, nombre, cuerpo, pq, trozos, fin=0, reanudar=False):
    """Escribe el resultado de <cuerpo> en OUTPUTS/gold/<nombre>/part_*.parquet.
    Devuelve True si la tabla quedo completa."""
    carpeta = os.path.join(pq, nombre)
    os.makedirs(carpeta, exist_ok=True)
    marca = os.path.join(carpeta, "_OK")
    if not reanudar:
        for viejo in glob.glob(os.path.join(carpeta, "*.parquet")):
            try:
                os.remove(viejo)
            except OSError:
                open(viejo, "wb").close()
    if os.path.exists(marca):
        os.remove(marca)

    # Las columnas LIST no viajan a Power BI: se serializan a texto.
    cols = con.execute(f"DESCRIBE SELECT * FROM ({cuerpo}) LIMIT 0").fetchall()
    proj = ", ".join(
        (f'CAST("{c[0]}" AS VARCHAR) AS "{c[0]}"'
         if "[]" in c[1] or c[1].upper().startswith(("LIST", "STRUCT", "MAP"))
         else f'"{c[0]}"')
        for c in cols)
    col0 = cols[0][0]

    for k in range(max(trozos, 1)):
        ruta = os.path.join(carpeta, f"part_{k:04d}.parquet")
        if reanudar and parquet_valido(ruta):
            continue
        if fin and time.time() > fin:
            return False
        filtro = ("" if trozos <= 1
                  else f' WHERE abs(hash("{col0}")) % {trozos} = {k}')
        con.execute(f"COPY (SELECT {proj} FROM ({cuerpo}){filtro}) "
                    f"TO '{ruta.replace(chr(92), '/')}' "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        if trozos > 1:
            print(f"    {nombre} trozo {k+1}/{trozos}", flush=True)

    open(marca, "w").close()
    return True


if __name__ == "__main__":
    main()
