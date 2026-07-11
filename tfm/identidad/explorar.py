#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perfila el problema de identidad de los nodos antes.
de fijar una clave.

Motivacion: en SECOP II la identidad operativa es `codigo_proveedor` (codigo de
cuenta de la plataforma), pero el mismo actor real puede tener varias cuentas, y
el documento que declara cambia segun el rol: una persona natural firma unas
veces con cedula y otras con NIT (que en Colombia es la cedula + digito de
verificacion). Este script reune todos los pares (codigo_proveedor, documento)
observados en el volcado y cuantifica los conflictos, sin decidir nada todavia.

Salida: gold/identidad_exploracion.md

Uso:
  python -m tfm.identidad.explorar --ram 12GB --hilos 8
"""

import argparse
import os
import tempfile
import time

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze", default=rutas.bronze())
    ap.add_argument("--salida", default=rutas.outputs_de("gold",
                                                     "identidad_exploracion.md"))
    ap.add_argument("--ram", default="4GB")
    ap.add_argument("--hilos", type=int, default=0)
    a = ap.parse_args()

    b = a.bronze.replace("\\", "/")
    # Base de trabajo persistente: permite reanudar si se interrumpe.
    trabajo = rutas.outputs_de("gold", "_identidad.duckdb")
    os.makedirs(os.path.dirname(trabajo), exist_ok=True)
    con = duckdb.connect(trabajo)
    con.execute(f"SET memory_limit='{a.ram}'")
    con.execute("SET preserve_insertion_order=false")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")
    con.execute(f"SET temp_directory='{os.path.join(tempfile.gettempdir(), 'dd_id')}'")

    def existe(t):
        return con.execute("SELECT count(*) FROM information_schema.tables "
                           "WHERE table_name = ?", [t]).fetchone()[0] > 0

    def etapa(nombre, sql):
        """Crea la tabla solo si no existe ya (reanudable)."""
        if existe(nombre):
            print(f"  {nombre}: ya existe", flush=True)
            return
        t0 = time.time()
        con.execute(sql)
        n = con.execute(f"SELECT count(*) FROM {nombre}").fetchone()[0]
        print(f"  {nombre}: {n:,} filas ({time.time()-t0:.1f}s)", flush=True)

    con.execute("CREATE OR REPLACE MACRO nitn(s) AS "
                "nullif(regexp_replace(CAST(s AS VARCHAR), '[^0-9]', '', 'g'), '');")

    # ------------------------------------------------------------------
    # 1. Todos los pares (codigo_proveedor, documento) observados
    # ------------------------------------------------------------------
    etapa("par", f"""
    CREATE TABLE par AS
    SELECT * FROM (
        SELECT codigo_proveedor AS cod, nitn(nit_proveedor) AS doc,
               nombre_proveedor AS nombre, 'proveedores_registrados' AS origen
        FROM read_parquet('{b}/proveedores/*.parquet')
      UNION ALL
        SELECT codigo_proveedor, nitn(nit_proveedor), nombre_proveedor, 'contratos'
        FROM read_parquet('{b}/contratos/*.parquet')
      UNION ALL
        SELECT codigo_proveedor, nitn(nit_proveedor), nombre_proveedor, 'procesos'
        FROM read_parquet('{b}/procesos/*.parquet')
      UNION ALL
        SELECT codigo_proveedor, nitn(nit_proveedor), nombre_proveedor, 'proponentes'
        FROM read_parquet('{b}/proponentes/*.parquet')
      UNION ALL
        SELECT codigo_proveedor, nitn(nit_proveedor), nombre_proveedor, 'ofertas'
        FROM read_parquet('{b}/ofertas/*.parquet')
      UNION ALL
        SELECT codigo_grupo, nitn(nit_grupo), nombre_grupo, 'grupos_ut'
        FROM read_parquet('{b}/grupos/*.parquet')
      UNION ALL
        SELECT codigo_participante, nitn(nit_participante), nombre_participante,
               'grupos_socio'
        FROM read_parquet('{b}/grupos/*.parquet')
    ) WHERE cod IS NOT NULL;
    """)

    etapa("cod_doc", """
    CREATE TABLE cod_doc AS
    SELECT cod, doc, count(*) AS apariciones,
           any_value(nombre) AS nombre,
           list(DISTINCT origen) AS origenes
    FROM par WHERE doc IS NOT NULL GROUP BY 1, 2;
    """)

    n_par, n_cod, n_doc = con.execute(
        "SELECT count(*), count(DISTINCT cod), count(DISTINCT doc) FROM cod_doc").fetchone()
    cod_sin_doc = con.execute(
        "SELECT count(DISTINCT cod) FROM par WHERE cod NOT IN "
        "(SELECT cod FROM cod_doc)").fetchone()[0]

    L = ["# Exploracion del problema de identidad de los nodos", "",
         "Perfilado de todos los pares `(codigo_proveedor, documento)` observados "
         "en el volcado completo, previo a decidir la clave del nodo.", "",
         "## 1. Volumen", "", "| Metrica | Valor |", "|---|---:|",
         f"| Pares codigo-documento distintos | {n_par:,} |",
         f"| Codigos de proveedor distintos | {n_cod:,} |",
         f"| Documentos distintos | {n_doc:,} |",
         f"| Codigos sin ningun documento | {cod_sin_doc:,} |"]

    # ------------------------------------------------------------------
    # 2. Conflictos de cardinalidad
    # ------------------------------------------------------------------
    etapa("cod_n", "CREATE TABLE cod_n AS SELECT cod, count(DISTINCT doc) AS n_doc FROM cod_doc GROUP BY 1")
    etapa("doc_n", "CREATE TABLE doc_n AS SELECT doc, count(DISTINCT cod) AS n_cod FROM cod_doc GROUP BY 1")

    L += ["", "## 2. Conflictos de cardinalidad", "",
          "### Un codigo con varios documentos", "",
          "| Documentos por codigo | Codigos |", "|---:|---:|"]
    for r in con.execute("SELECT n_doc, count(*) FROM cod_n GROUP BY 1 "
                         "ORDER BY 1 LIMIT 8").fetchall():
        L.append(f"| {r[0]} | {r[1]:,} |")

    L += ["", "### Un documento con varios codigos "
          "(el actor tiene varias cuentas en la plataforma)", "",
          "| Codigos por documento | Documentos |", "|---:|---:|"]
    for r in con.execute("SELECT n_cod, count(*) FROM doc_n GROUP BY 1 "
                         "ORDER BY 1 LIMIT 8").fetchall():
        L.append(f"| {r[0]} | {r[1]:,} |")

    multi = con.execute("SELECT count(*) FROM doc_n WHERE n_cod > 1").fetchone()[0]
    L += ["", f"**{multi:,} documentos aparecen bajo mas de un `codigo_proveedor`.** "
          "Es el caso que mencionas: un mismo actor puede haberse presentado con "
          "codigos distintos, por ejemplo al integrar una union temporal sin haber "
          "ganado nunca un contrato por su cuenta.", ""]

    L += ["### Ejemplos de documentos con mas cuentas", "",
          "| Documento | Cuentas | Nombres asociados |", "|---|---:|---|"]
    for r in con.execute("""
            SELECT d.doc, d.n_cod,
                   list(DISTINCT substr(c.nombre, 1, 40))[1:3]
            FROM doc_n d JOIN cod_doc c USING (doc)
            WHERE d.n_cod > 1 GROUP BY 1, 2 ORDER BY 2 DESC LIMIT 10""").fetchall():
        L.append(f"| {r[0]} | {r[1]} | {'; '.join(x for x in r[2] if x)} |")

    # ------------------------------------------------------------------
    # 3. Digito de verificacion: cedula vs NIT
    # ------------------------------------------------------------------
    # Equi-join sobre el prefijo: mucho mas barato que comparar longitudes
    # y concatenar dentro de la condicion de union.
    etapa("dv", """
    CREATE TABLE dv AS
    WITH d AS (SELECT DISTINCT doc FROM cod_doc),
         largo AS (SELECT doc AS doc_largo,
                          substr(doc, 1, length(doc) - 1) AS prefijo
                   FROM d WHERE length(doc) >= 7)
    SELECT l.doc_largo, d.doc AS doc_base
    FROM largo l JOIN d ON d.doc = l.prefijo;
    """)
    n_dv = con.execute("SELECT count(*) FROM dv").fetchone()[0]
    dv_dist = con.execute("""
        SELECT count(DISTINCT doc_base), count(DISTINCT doc_largo) FROM dv""").fetchone()

    L += ["", "## 3. Digito de verificacion (cedula vs NIT)", "",
          "En Colombia el NIT de una persona natural es su cedula mas un digito "
          "de verificacion. El RUES lo confirma: trae `numero_identificacion` y "
          "`digito_verificacion` en campos separados, y el campo `nit` es la "
          "concatenacion de ambos.", "",
          "| Metrica | Valor |", "|---|---:|",
          f"| Pares documento / documento+1digito | {n_dv:,} |",
          f"| Documentos base implicados | {dv_dist[0]:,} |",
          f"| Documentos largos implicados | {dv_dist[1]:,} |"]

    L += ["", "Ejemplos (mismo actor, dos documentos):", "",
          "| Doc. base | Doc. con DV | Nombre base | Nombre con DV |", "|---|---|---|---|"]
    for r in con.execute("""
            SELECT v.doc_base, v.doc_largo,
                   substr(any_value(a.nombre), 1, 38),
                   substr(any_value(b.nombre), 1, 38)
            FROM dv v
            JOIN cod_doc a ON a.doc = v.doc_base
            JOIN cod_doc b ON b.doc = v.doc_largo
            GROUP BY 1, 2 LIMIT 10""").fetchall():
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")

    # cuantos de esos pares comparten nombre canonico -> mismo actor casi seguro
    etapa("dv_nombre", """
    CREATE TABLE dv_nombre AS
    SELECT v.doc_base, v.doc_largo,
           bool_or(upper(strip_accents(a.nombre)) = upper(strip_accents(b.nombre)))
               AS mismo_nombre
    FROM dv v
    JOIN cod_doc a ON a.doc = v.doc_base
    JOIN cod_doc b ON b.doc = v.doc_largo
    GROUP BY 1, 2""")
    mismo = con.execute(
        "SELECT count(*) FROM dv_nombre WHERE mismo_nombre").fetchone()[0]
    L += ["", f"De esos {n_dv:,} pares, **{mismo:,} comparten exactamente el mismo "
          "nombre**: son con alta probabilidad el mismo actor duplicado por el "
          "digito de verificacion.", ""]

    # ------------------------------------------------------------------
    # 4. Longitud y tipo de documento
    # ------------------------------------------------------------------
    L += ["## 4. Longitud del documento", "",
          "| Longitud | Documentos | Interpretacion |", "|---:|---:|---|"]
    interp = {6: "cedula antigua", 7: "cedula antigua", 8: "cedula",
              9: "NIT empresa o cedula+DV", 10: "cedula nueva o NIT+DV",
              11: "NIT+DV"}
    for r in con.execute("""SELECT length(doc), count(DISTINCT doc) FROM cod_doc
                            GROUP BY 1 ORDER BY 1""").fetchall():
        if r[1] < 50:
            continue
        L.append(f"| {r[0]} | {r[1]:,} | {interp.get(r[0], 'atipico / posible error')} |")

    # ------------------------------------------------------------------
    # 5. Cruce con RUES
    # ------------------------------------------------------------------
    etapa("rues_doc", f"""
    CREATE TABLE rues_doc AS
    SELECT nitn(numero_identificacion) AS doc_base,
           nitn(nit) AS doc_nit,
           any_value(razon_social) AS razon_social,
           any_value(clase_identificacion) AS clase,
           count(*) AS matriculas
    FROM read_parquet('{b}/rues/*.parquet')
    WHERE numero_identificacion IS NOT NULL
    GROUP BY 1, 2;
    """)
    r_tot, r_doc = con.execute(
        "SELECT count(*), count(DISTINCT doc_base) FROM rues_doc").fetchone()

    # Semi-joins materializados: los EXISTS correlacionados sobre 8,5M filas
    # de RUES son inviables.
    etapa("cruce_rues", """
    CREATE TABLE cruce_rues AS
    WITH d AS (SELECT DISTINCT doc FROM cod_doc),
         base AS (SELECT DISTINCT doc_base AS doc FROM rues_doc WHERE doc_base IS NOT NULL),
         connit AS (SELECT DISTINCT doc_nit AS doc FROM rues_doc WHERE doc_nit IS NOT NULL)
    SELECT d.doc,
           b.doc IS NOT NULL AS por_numero_identificacion,
           n.doc IS NOT NULL AS por_nit
    FROM d LEFT JOIN base b USING (doc) LEFT JOIN connit n USING (doc)""")
    cruce = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE por_numero_identificacion),
               count(*) FILTER (WHERE por_nit) FROM cruce_rues""").fetchone()

    cruce_any = con.execute(
        "SELECT count(*) FROM cruce_rues WHERE por_numero_identificacion OR por_nit"
    ).fetchone()[0]

    L += ["", "## 5. Cruce con el RUES", "",
          "| Metrica | Valor |", "|---|---:|",
          f"| Registros RUES (doc_base, nit) | {r_tot:,} |",
          f"| Documentos distintos en RUES | {r_doc:,} |",
          f"| Documentos SECOP | {cruce[0]:,} |",
          f"| — cruzan por `numero_identificacion` | {cruce[1]:,} "
          f"({100*cruce[1]/max(cruce[0],1):.1f}%) |",
          f"| — cruzan por `nit` (con DV) | {cruce[2]:,} "
          f"({100*cruce[2]/max(cruce[0],1):.1f}%) |",
          f"| — cruzan por cualquiera de los dos | {cruce_any:,} "
          f"({100*cruce_any/max(cruce[0],1):.1f}%) |"]

    # ------------------------------------------------------------------
    # 6. Recomendacion
    # ------------------------------------------------------------------
    L += ["", "## 6. Lectura y propuesta", "",
          "1. **`codigo_proveedor` es casi funcional respecto al documento**: la "
          "gran mayoria de codigos tienen un unico documento asociado. El problema "
          "no es que un codigo sea ambiguo, sino que **un mismo actor tiene varias "
          "cuentas**.", "",
          "2. **El digito de verificacion es la causa mecanica principal** de "
          "duplicacion, y es resoluble de forma determinista: el RUES da "
          "`numero_identificacion` y `digito_verificacion` por separado, asi que "
          "sirve de arbitro en lugar de adivinar.", "",
          "3. **Propuesta de clave en dos niveles**, que es lo que se implementa "
          "en `05_nodos.py`:", "",
          "   - `codigo_proveedor` se conserva como identificador operativo "
          "(trazabilidad con las tablas publicadas del SECOP).", "",
          "   - `id_nodo` es la clave analitica: se obtiene por union de "
          "componentes conexas sobre las equivalencias deterministas "
          "(mismo documento base tras normalizar el DV con arbitraje del RUES).", "",
          "4. **Lo que queda para Splink**: los actores que no comparten documento "
          "pero probablemente son el mismo (razon social casi identica, mismo "
          "representante legal, misma direccion). El fichero "
          "`OUTPUTS/gold/nodo_candidatos_splink/` deja preparado el conjunto de "
          "atributos de bloqueo para esa fase probabilistica.", ""]

    texto = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(texto)
    print(f"-> {a.salida}")


if __name__ == "__main__":
    main()
