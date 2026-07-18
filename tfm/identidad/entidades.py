#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolida las entidades estatales duplicadas.

Problema: `codigo_entidad` es unico (27.114 de 27.114), pero NO identifica a la
entidad real. Un mismo NIT se reparte legitimamente entre muchas unidades
ejecutoras con nombres distintos (el SENA tiene 135 codigos, cada uno una
regional), y eso hay que respetarlo. Lo que si es duplicacion es que coincidan
NIT y nombre: 782 grupos, entre ellos 37 codigos que el propio SECOP marca con
sufijo `_DUP3`, `_DUP8`, etc.

Salida (OUTPUTS/gold/ y OUTPUTS/gold/secop_tfm_red2025.duckdb):
    entidad_resuelta        codigo_entidad -> id_entidad, con la regla aplicada
    entidad_evidencia       el antes y el despues, grupo a grupo
    nodo_entidad_consolidado  nodos de entidad ya agrupados

Uso:
    python -m tfm.identidad.entidades --ram 12GB --hilos 8
"""

import argparse
import glob
import os
import tempfile

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()

SQL = r"""
CREATE OR REPLACE MACRO norm(s) AS
    nullif(trim(regexp_replace(upper(strip_accents(CAST(s AS VARCHAR))),
                               '\s+', ' ', 'g')), '');

-- ---------------------------------------------------------------------------
-- 1. Punto de partida: como estan las entidades en bruto
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE entidad_bruto AS
SELECT
    e.codigo_entidad,
    e.nit_entidad,
    e.nombre_entidad,
    e.nombre_entidad_norm,
    e.orden_entidad, e.sector, e.rama, e.entidad_centralizada,
    e.tipo_entidad, e.departamento_norm, e.ciudad,
    e.n_contratos_hist, e.n_procesos_hist,
    -- señal 1: el propio SECOP marca el duplicado en el codigo
    regexp_matches(e.codigo_entidad, '_DUP[0-9]*$')          AS marcado_dup,
    regexp_extract(e.codigo_entidad, '^([^_]+)')             AS codigo_base
FROM dim_entidad e;


-- ---------------------------------------------------------------------------
-- 2. Reglas de consolidacion, en orden de fuerza.
--    R1  sufijo _DUP y el codigo base existe          -> duplicado explicito
--    R2  mismo NIT y mismo nombre normalizado         -> duplicado por contenido
--    R3  todo lo demas                                -> unidad ejecutora propia
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE entidad_resuelta AS
WITH r1 AS (
    SELECT b.codigo_entidad, b.codigo_base AS destino, 'R1 sufijo _DUP' AS regla
    FROM entidad_bruto b
    WHERE b.marcado_dup
      AND EXISTS (SELECT 1 FROM entidad_bruto x WHERE x.codigo_entidad = b.codigo_base)
),
-- Representante del grupo (NIT, nombre): el codigo con mas actividad, y a
-- igualdad el menor, para que el resultado sea estable entre ejecuciones.
r2 AS (
    SELECT codigo_entidad,
           first(codigo_entidad) OVER (
               PARTITION BY nit_entidad, nombre_entidad_norm
               ORDER BY n_contratos_hist DESC, n_procesos_hist DESC, codigo_entidad
           ) AS destino,
           'R2 mismo NIT y nombre' AS regla
    FROM entidad_bruto
    WHERE nit_entidad IS NOT NULL AND nombre_entidad_norm IS NOT NULL
)
SELECT
    b.codigo_entidad,
    coalesce(r1.destino, r2.destino, b.codigo_entidad)       AS codigo_entidad_canonico,
    'ENT:' || coalesce(r1.destino, r2.destino, b.codigo_entidad) AS id_entidad,
    CASE WHEN r1.destino IS NOT NULL THEN r1.regla
         WHEN r2.destino IS NOT NULL AND r2.destino <> b.codigo_entidad THEN r2.regla
         ELSE 'R3 sin cambio' END                             AS regla,
    b.nit_entidad,
    b.nombre_entidad,
    b.nombre_entidad_norm
FROM entidad_bruto b
LEFT JOIN r1 ON r1.codigo_entidad = b.codigo_entidad
LEFT JOIN r2 ON r2.codigo_entidad = b.codigo_entidad;


-- ---------------------------------------------------------------------------
-- 3. Evidencia: el antes y el despues, grupo a grupo
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE entidad_evidencia AS
SELECT
    r.id_entidad,
    r.codigo_entidad_canonico,
    any_value(r.nit_entidad)                                 AS nit_entidad,
    count(*)                                                 AS codigos_antes,
    1                                                        AS entidades_despues,
    list(r.codigo_entidad ORDER BY r.codigo_entidad)         AS codigos_agrupados,
    list(DISTINCT r.nombre_entidad)                          AS nombres_observados,
    count(DISTINCT r.nombre_entidad)                         AS n_nombres_distintos,
    list(DISTINCT r.regla)                                   AS reglas
FROM entidad_resuelta r
GROUP BY 1, 2
HAVING count(*) > 1;


-- ---------------------------------------------------------------------------
-- 4. Comprobacion de la premisa: el NIT se repite, el codigo no
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE entidad_diagnostico AS
SELECT 'codigos_entidad'            AS metrica, count(*)                          AS valor FROM entidad_bruto
UNION ALL SELECT 'codigos distintos',        count(DISTINCT codigo_entidad)        FROM entidad_bruto
UNION ALL SELECT 'nits distintos',           count(DISTINCT nit_entidad)           FROM entidad_bruto
UNION ALL SELECT 'nombres normalizados distintos', count(DISTINCT nombre_entidad_norm) FROM entidad_bruto
UNION ALL SELECT 'nits en mas de un codigo',
    (SELECT count(*) FROM (SELECT nit_entidad FROM entidad_bruto WHERE nit_entidad IS NOT NULL
                           GROUP BY 1 HAVING count(DISTINCT codigo_entidad) > 1))
UNION ALL SELECT 'grupos NIT+nombre duplicados',
    (SELECT count(*) FROM (SELECT nit_entidad, nombre_entidad_norm FROM entidad_bruto
                           WHERE nit_entidad IS NOT NULL GROUP BY 1, 2
                           HAVING count(DISTINCT codigo_entidad) > 1))
UNION ALL SELECT 'codigos con sufijo _DUP',
    (SELECT count(*) FROM entidad_bruto WHERE marcado_dup)
UNION ALL SELECT 'entidades despues de consolidar',
    (SELECT count(DISTINCT id_entidad) FROM entidad_resuelta);


-- ---------------------------------------------------------------------------
-- 5. Nodos de entidad ya consolidados
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TABLE nodo_entidad_consolidado AS
SELECT
    r.id_entidad,
    r.codigo_entidad_canonico,
    count(DISTINCT r.codigo_entidad)              AS n_codigos_fusionados,
    list(DISTINCT r.codigo_entidad)               AS codigos_entidad,
    any_value(b.nit_entidad)                      AS nit_entidad,
    max(b.nombre_entidad)                         AS nombre,
    max(b.nombre_entidad_norm)                    AS nombre_canonico,
    max(b.orden_entidad)                          AS orden_entidad,
    max(b.sector)                                 AS sector,
    max(b.rama)                                   AS rama,
    max(b.entidad_centralizada)                   AS entidad_centralizada,
    max(b.tipo_entidad)                           AS tipo_entidad,
    max(b.departamento_norm)                      AS departamento_norm,
    max(b.ciudad)                                 AS ciudad,
    sum(n.n_contratos_2025)                       AS n_contratos_2025,
    sum(n.valor_contratos_2025)                   AS valor_contratos_2025,
    sum(n.n_proveedores_2025)                     AS n_proveedores_2025
FROM entidad_resuelta r
JOIN entidad_bruto b ON b.codigo_entidad = r.codigo_entidad
LEFT JOIN nodo_entidad n ON n.codigo_entidad = r.codigo_entidad
GROUP BY 1, 2;
"""


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
    return sorted(f for f in glob.glob(os.path.join(carpeta, "*.parquet"))
                  if parquet_valido(f))


def lista_sql(fs):
    return "[" + ", ".join("'" + f.replace("\\", "/").replace("'", "''") + "'"
                           for f in fs) + "]"


PUBLICAR = ["entidad_resuelta", "entidad_evidencia", "entidad_diagnostico",
            "nodo_entidad_consolidado"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ)
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    a = ap.parse_args()

    gold = os.path.join(a.raiz, "gold")
    pq = os.path.join(gold, "parquet")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.ram}'")
    con.execute("SET preserve_insertion_order=false")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")
    tmp = os.path.join(tempfile.gettempdir(), "dd_ent")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")

    for t in ("dim_entidad", "nodo_entidad"):
        fs = parquets(os.path.join(pq, t))
        if not fs:
            raise SystemExit(f"falta OUTPUTS/gold/{t}")
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                    f"read_parquet({lista_sql(fs)}, union_by_name=true)")

    con.execute(SQL)

    print("[diagnostico]")
    for m, v in con.execute("SELECT * FROM entidad_diagnostico").fetchall():
        print(f"  {m:<34} {v:>10,}")

    print("\n[consolidacion]")
    print(con.execute("""SELECT regla, count(*) AS codigos
                         FROM entidad_resuelta GROUP BY 1 ORDER BY 2 DESC""").df()
          .to_string(index=False))

    print("\n[grupos con mas codigos fusionados]")
    print(con.execute("""
        SELECT codigo_entidad_canonico, nit_entidad, codigos_antes,
               n_nombres_distintos, nombres_observados[1] AS nombre
        FROM entidad_evidencia ORDER BY codigos_antes DESC LIMIT 8""").df()
          .to_string(index=False))

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
    con.close()

    # registrar las nuevas tablas en la base de red
    bd = os.path.join(gold, "secop_tfm_red2025.duckdb")
    out = duckdb.connect(bd)
    for t in PUBLICAR:
        fs = parquets(os.path.join(pq, t))
        out.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                    f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    # vinculos con la entidad ya consolidada
    out.execute("""
        CREATE OR REPLACE VIEW v_vinculo_entidad_consolidada AS
        SELECT v.tipo_vinculo,
               coalesce(eo.id_entidad, v.origen)  AS origen,
               coalesce(ed.id_entidad, v.destino) AS destino,
               sum(v.peso) AS peso, sum(v.valor) AS valor
        FROM vinculo_2025 v
        LEFT JOIN entidad_resuelta eo ON 'ENT:' || eo.codigo_entidad = v.origen
        LEFT JOIN entidad_resuelta ed ON 'ENT:' || ed.codigo_entidad = v.destino
        GROUP BY 1, 2, 3""")
    out.close()
    print(f"\nOK -> {pq} y {bd}")


if __name__ == "__main__":
    main()
