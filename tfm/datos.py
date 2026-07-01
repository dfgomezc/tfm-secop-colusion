#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceso al modelo dimensional.

`conectar()` devuelve una conexión DuckDB en memoria con los parquet montados
como vistas y las vistas analíticas de `tfm/ingesta/gold.sql` reconstruidas
encima.

Se trabaja siempre sobre los parquet y no sobre `gold/secop_tfm.duckdb`: ese
fichero guarda las rutas absolutas de la máquina que lo creó y no resuelve al
abrirlo en otra.

    from tfm import datos
    con = datos.conectar()
    con.execute("SELECT count(*) FROM fact_proceso").fetchone()
"""

import glob
import re
import os

from tfm import rutas


def tablas():
    """Las tablas disponibles en el volcado, por nombre de carpeta."""
    base = rutas.parquet()
    if not os.path.isdir(base):
        return []
    return sorted(os.path.basename(c) for c in glob.glob(os.path.join(base, "*"))
                  if os.path.isdir(c) and glob.glob(os.path.join(c, "*.parquet")))


def conectar(memoria="6GB", solo=None):
    """Conexión en memoria con el modelo dimensional montado como vistas.

    `solo` limita el montaje a las tablas indicadas, que ahorra tiempo cuando
    una consulta necesita dos de las treinta y cuatro.

    Si no hay parquet que montar se levanta un error explícito, en vez de
    dejar que la consulta falle más tarde con «tabla no encontrada».
    """
    import duckdb

    disponibles = tablas()
    if not disponibles:
        raise FileNotFoundError(
            f"no hay ningún parquet en {rutas.parquet()}.\n"
            f"Descomprime el paquete de datos según el README, o exporta "
            f"TFM_OUTPUTS apuntando a donde estén.")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memoria}'")
    for nombre in disponibles:
        if solo and nombre not in solo:
            continue
        patron = os.path.join(rutas.parquet(), nombre, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW {nombre} AS "
                    f"SELECT * FROM read_parquet('{patron.replace(os.sep, '/')}')")
    if not solo:
        vistas_analiticas(con)
    return con


def vistas_analiticas(con):
    """Monta las vistas analíticas declaradas en `gold.sql`.

    El índice de riesgo, las aristas del grafo, los triángulos y las señales de
    contacto compartido son vistas, no tablas materializadas, de modo que hay
    que recrearlas sobre la conexión. Se leen del propio SQL para no duplicar
    su definición. Las que dependan de una tabla ausente se omiten, lo que
    permite trabajar con un volcado parcial.
    """
    ruta_sql = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ingesta", "gold.sql")
    if not os.path.exists(ruta_sql):
        return []
    sql = open(ruta_sql, encoding="utf-8").read()
    #: Los comentarios se retiran antes de partir por punto y coma: `gold.sql`
    #: contiene comentarios con `;`, que si no romperían las sentencias.
    sql = re.sub(r"--[^\n]*", "", sql)
    creadas = []
    for sentencia in sql.split(";"):
        if not re.search(r"CREATE\s+(OR\s+REPLACE\s+)?VIEW", sentencia, re.I):
            continue
        try:
            con.execute(sentencia)
            m = re.search(r"VIEW\s+(\w+)", sentencia, re.I)
            creadas.append(m.group(1) if m else "?")
        except Exception:
            continue
    return creadas


def montadas(con):
    """Las vistas montadas sobre la conexión."""
    return [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "ORDER BY table_name").fetchall()]
