#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construye el universo de red: contratos suscritos en 2025.
sin prestacion de servicios, con tabla de nodos (SECOP + RUES) y de vinculos.

Ejecuta `nodos.sql` sentencia a sentencia sobre una base de trabajo
persistente (reanudable) y publica el resultado en OUTPUTS/gold/ y en
OUTPUTS/gold/secop_tfm_red2025.duckdb.

Uso:
  python -m tfm.identidad.nodos --ram 12GB --hilos 8
  python -m tfm.identidad.nodos --reanudar --limite-seg 300
  python -m tfm.identidad.nodos --rehacer nodo_2025 vinculo_2025
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
RAIZ = rutas.raiz()

# Tablas del modelo dimensional que hacen falta.
DESDE_GOLD = ["fact_contrato", "fact_proceso", "fact_proponente", "fact_oferta",
              "dim_proveedor", "dim_entidad", "dim_modalidad", "bridge_ut_socio"]

PUBLICAR = ["contrato_2025", "proceso_2025", "participacion_2025",
            "proceso_competencia", "nodo_2025", "nodo_cuenta",
            "nodo_representante", "nodo_entidad", "vinculo_2025",
            "nodo_candidato_splink"]


def parquet_valido(ruta):
    try:
        if os.path.getsize(ruta) < 8:
            return False
        with open(ruta, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def lista_sql(fs):
    return "[" + ", ".join("'" + f.replace("\\", "/").replace("'", "''") + "'"
                           for f in fs) + "]"


def parquets(carpeta):
    return sorted(f for f in glob.glob(os.path.join(carpeta, "*.parquet"))
                  if parquet_valido(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ)
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--limite-seg", type=int, default=0, dest="limite")
    ap.add_argument("--rehacer", nargs="*", default=[],
                    help="tablas a reconstruir aunque ya existan")
    a = ap.parse_args()

    bronze = os.path.join(a.raiz, "bronze")
    gold = os.path.join(a.raiz, "gold")
    pq = os.path.join(gold, "parquet")
    trabajo = os.path.join(gold, "_red2025.duckdb")
    os.makedirs(pq, exist_ok=True)

    if not a.reanudar and os.path.exists(trabajo):
        for f in (trabajo, trabajo + ".wal"):
            if os.path.exists(f):
                os.remove(f)

    con = duckdb.connect(trabajo)
    con.execute(f"SET memory_limit='{a.ram}'")
    con.execute("SET preserve_insertion_order=false")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")
    tmp = os.path.join(tempfile.gettempdir(), "dd_red")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")

    # ---- fuentes -------------------------------------------------------
    print("[fuentes]")
    fs = parquets(os.path.join(bronze, "rues"))
    con.execute(f"CREATE OR REPLACE VIEW stg_rues AS SELECT * FROM "
                f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    print(f"  stg_rues ({len(fs)} ficheros)")
    for t in DESDE_GOLD:
        fs = parquets(os.path.join(pq, t))
        if not fs:
            raise SystemExit(f"falta OUTPUTS/gold/{t}: ejecuta antes 02_gold.py")
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                    f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    print(f"  {len(DESDE_GOLD)} tablas del modelo dimensional")

    # ---- modelo --------------------------------------------------------
    print("\n[red 2025]")
    with open(os.path.join(AQUI, "nodos.sql"), "r", encoding="utf-8") as fh:
        sentencias = [s for s in fh.read().split(";\n")
                      if re.search(r"\bCREATE\b", s, re.I)]

    fin = (time.time() + a.limite) if a.limite else 0
    pendientes = 0
    for s in sentencias:
        m = re.search(r"(TABLE|VIEW|MACRO)\s+(?:IF NOT EXISTS\s+)?([A-Za-z_0-9]+)",
                      s, re.I)
        objeto = m.group(2) if m else "?"
        clase = m.group(1).upper() if m else ""

        if clase == "MACRO":
            con.execute(s + ";")
            continue

        existe = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [objeto]).fetchone()[0] > 0
        if existe and objeto not in a.rehacer and clase == "TABLE":
            n = con.execute(f"SELECT count(*) FROM {objeto}").fetchone()[0]
            print(f"  {objeto:<24} {n:>12,}  (ya existe)")
            continue
        if fin and time.time() > fin:
            pendientes += 1
            continue

        t0 = time.time()
        con.execute(s + ";")
        n = con.execute(f"SELECT count(*) FROM {objeto}").fetchone()[0]
        print(f"  {objeto:<24} {n:>12,}  ({time.time()-t0:.1f}s)", flush=True)

    if pendientes:
        print(f"\n[!] quedan {pendientes} objetos; relanza con --reanudar")
        con.close()
        return

    # ---- publicar ------------------------------------------------------
    print("\n[publicar]")
    for t in PUBLICAR:
        destino = os.path.join(pq, t)
        os.makedirs(destino, exist_ok=True)
        for viejo in glob.glob(os.path.join(destino, "*.parquet")):
            os.remove(viejo)
        cols = con.execute(f"DESCRIBE {t}").fetchall()
        proj = ", ".join(
            (f'CAST("{c[0]}" AS VARCHAR) AS "{c[0]}"'
             if "[]" in c[1] or c[1].upper().startswith(("LIST", "STRUCT", "MAP"))
             else f'"{c[0]}"') for c in cols)
        ruta = os.path.join(destino, "part_0000.parquet").replace("\\", "/")
        con.execute(f"COPY (SELECT {proj} FROM {t}) TO '{ruta}' "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"  {t}.parquet")
    con.close()

    # ---- base ligera de consulta ---------------------------------------
    bd = os.path.join(gold, "secop_tfm_red2025.duckdb")
    for f in (bd, bd + ".wal"):
        if os.path.exists(f):
            os.remove(f)
    out = duckdb.connect(bd)
    for t in PUBLICAR:
        fs = parquets(os.path.join(pq, t))
        out.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                    f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    out.execute("""
        CREATE OR REPLACE VIEW v_triangulo_2025 AS
        WITH adjudica AS (SELECT origen AS entidad, destino AS nodo,
                                 peso AS contratos, valor, detalle
                          FROM vinculo_2025 WHERE tipo_vinculo = 'adjudica'),
             integra  AS (SELECT origen AS nodo_ut, destino AS nodo_socio
                          FROM vinculo_2025 WHERE tipo_vinculo = 'integra')
        SELECT a.entidad, a.nodo AS nodo_ut, i.nodo_socio,
               a.contratos AS contratos_ut, a.valor AS valor_ut,
               d.contratos AS contratos_directos_socio,
               d.valor AS valor_directo_socio,
               d.detalle AS modalidad_socio
        FROM adjudica a
        JOIN integra  i ON i.nodo_ut = a.nodo
        JOIN adjudica d ON d.entidad = a.entidad AND d.nodo = i.nodo_socio""")
    out.close()
    print(f"\nOK\n  datos : {pq}\n  base  : {bd}")


if __name__ == "__main__":
    main()
