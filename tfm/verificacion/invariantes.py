#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprueba que el modelo esta completo y es coherente.

No genera nada: solo lee y valida. Sirve para saber en un minuto si lo que hay
en gold/ esta bien, sin volver a ejecutar el pipeline entero.

Uso:
    python -m tfm.verificacion.invariantes
    python -m tfm.verificacion.invariantes --detalle       # ademas imprime una fila de ejemplo
"""

import argparse
import glob
import os
import sys

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()

# tabla -> (filas esperadas, tolerancia relativa)
ESPERADO = {
    "dim_tiempo":                 (4_383, 0.30),
    "dim_departamento":             (357, 0.30),
    "dim_entidad":               (27_114, 0.10),
    "dim_proveedor":          (1_849_655, 0.10),
    "dim_categoria":             (19_666, 0.10),
    "dim_modalidad":                 (17, 0.50),
    "dim_tipo_contrato":             (25, 0.50),
    "fact_contrato":          (5_859_813, 0.05),
    "fact_proceso":           (8_438_722, 0.05),
    "fact_proponente":        (2_219_036, 0.05),
    "fact_oferta":              (861_642, 0.05),
    "fact_sancion":                 (544, 0.20),
    "bridge_ut_socio":          (674_509, 0.05),
    "contrato_2025":            (223_987, 0.02),
    "proceso_2025":             (245_182, 0.02),
    "participacion_2025":       (963_566, 0.02),
    "proceso_competencia":      (217_363, 0.02),
    "nodo_2025":                (139_548, 0.02),
    "nodo_cuenta":              (146_565, 0.02),
    "nodo_entidad":               (3_872, 0.10),
    "nodo_representante":        (98_027, 0.10),
    "vinculo_2025":           (2_914_940, 0.02),
    "entidad_resuelta":          (27_114, 0.10),
    "nodo_entidad_consolidado":  (26_127, 0.10),
    "splink_cluster":            (91_231, 0.05),
    "splink_evidencia":           (2_348, 0.30),
    "splink_grupo_empresarial":   (1_162, 0.40),
}


def main():
    ap = argparse.ArgumentParser()
    #: El modelo dimensional vive en `OUTPUTS/gold`, que es lo que se
    #: descomprime del paquete de datos. Se mantiene `--raiz` para poder
    #: apuntar a un volcado distinto sin tocar variables de entorno.
    ap.add_argument("--raiz", default=rutas.parquet())
    ap.add_argument("--ram", default="4GB")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--detalle", action="store_true")
    a = ap.parse_args()

    pq = a.raiz
    fallos, avisos = [], []

    print("=" * 68)
    print("VERIFICACION DEL MODELO")
    print("=" * 68)

    # --- 1. ficheros presentes -------------------------------------------
    print("\n[1] Tablas publicadas en OUTPUTS/gold")
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.ram}'")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")

    for tabla, (esperadas, tol) in ESPERADO.items():
        carpeta = os.path.join(pq, tabla)
        fs = sorted(glob.glob(os.path.join(carpeta, "*.parquet")))
        fs = [f for f in fs if os.path.getsize(f) > 0]
        if not fs:
            print(f"  FALTA    {tabla}")
            fallos.append(f"falta la tabla {tabla}")
            continue
        lst = "[" + ", ".join("'" + f.replace("\\", "/") + "'" for f in fs) + "]"
        try:
            n = con.execute(f"SELECT sum(num_rows) FROM parquet_file_metadata({lst})").fetchone()[0]
        except Exception as exc:
            print(f"  CORRUPTA {tabla}: {str(exc)[:50]}")
            fallos.append(f"{tabla} ilegible")
            continue
        desvio = abs(n - esperadas) / max(esperadas, 1)
        if desvio <= tol:
            print(f"  ok       {tabla:<26} {n:>12,}")
        else:
            print(f"  DISTINTO {tabla:<26} {n:>12,}  (esperaba ~{esperadas:,})")
            avisos.append(f"{tabla}: {n:,} frente a ~{esperadas:,}")

    # --- 1 bis. vistas sobre el parquet -----------------------------------
    #: Las bases `.duckdb` guardan vistas cuyas rutas son absolutas y de la
    #: máquina que las creó, así que no resuelven en ningún otro sitio. El
    #: parquet sí es portátil, y es lo que `MAPA_REPOSITORIO.md` declara como
    #: fuente de verdad. Se monta una vista por tabla para que las
    #: comprobaciones de coherencia consulten el parquet directamente.
    pqr = pq.replace("\\", "/")
    montadas = set()
    for carpeta in sorted(glob.glob(os.path.join(pq, "*"))):
        nombre = os.path.basename(carpeta)
        if not glob.glob(os.path.join(carpeta, "*.parquet")):
            continue
        con.execute(f"CREATE OR REPLACE VIEW {nombre} AS "
                    f"SELECT * FROM read_parquet('{pqr}/{nombre}/*.parquet')")
        montadas.add(nombre)
    print(f"\n[1 bis] {len(montadas)} tablas montadas desde parquet")

    # --- 2. bases de consulta --------------------------------------------
    #: Los ficheros DuckDB son una comodidad local y no viajan con el paquete
    #: de datos: sus vistas guardan rutas absolutas y no resuelven en otra
    #: máquina. Su ausencia se informa, no se cuenta como fallo; quien decide
    #: sobre la integridad son los apartados [3] a [5], que consultan parquet.
    print("\n[2] Bases DuckDB (opcionales)")
    bases = {
        "secop_tfm.duckdb": ["fact_contrato", "fact_proceso", "v_cri_proveedor"],
        "secop_tfm_red2025.duckdb": ["nodo_2025", "vinculo_2025", "v_triangulo_2025"],
    }
    for nombre, objetos in bases.items():
        ruta = os.path.join(a.raiz, nombre)
        if not os.path.exists(ruta):
            print(f"  ausente  {nombre} (se consulta el parquet)")
            continue
        try:
            c = duckdb.connect(ruta, read_only=True)
            c.execute(f"SET memory_limit='{a.ram}'")
            for o in objetos:
                c.execute(f"SELECT * FROM {o} LIMIT 1").fetchall()
            c.close()
            print(f"  ok       {nombre}")
        except Exception as exc:
            #: Las vistas de estas bases apuntan al parquet por ruta absoluta.
            #: Fuera de la máquina que las creó no resuelven, y eso no dice
            #: nada sobre la integridad del modelo: los apartados [3] a [5]
            #: consultan el parquet y son los que deciden.
            print(f"  aviso    {nombre}: vistas con ruta absoluta, "
                  f"no resuelven aquí")
            avisos.append(f"{nombre}: vistas con ruta absoluta")

    # --- 3. coherencia ----------------------------------------------------
    print("\n[3] Coherencia del modelo dimensional")
    if {"fact_contrato", "dim_entidad", "dim_proveedor"} <= montadas:
        c = con
        pruebas = [
            ("claves unicas en fact_contrato",
             "SELECT count(*) - count(DISTINCT id_contrato) FROM fact_contrato"),
            ("claves unicas en fact_proceso",
             "SELECT count(*) - count(DISTINCT id_del_proceso) FROM fact_proceso"),
            ("claves unicas en dim_entidad",
             "SELECT count(*) - count(DISTINCT codigo_entidad) FROM dim_entidad"),
            ("claves unicas en dim_proveedor",
             "SELECT count(*) - count(DISTINCT codigo_proveedor) FROM dim_proveedor"),
            ("sin entidades huerfanas en fact_contrato",
             """SELECT count(*) FROM fact_contrato f WHERE f.codigo_entidad IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM dim_entidad d
                                WHERE d.codigo_entidad = f.codigo_entidad)"""),
            ("sin proveedores huerfanos en fact_contrato",
             """SELECT count(*) FROM fact_contrato f WHERE f.codigo_proveedor IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM dim_proveedor d
                                WHERE d.codigo_proveedor = f.codigo_proveedor)"""),
            #: La vista `v_competencia_proceso` vive en la base y no en el
            #: parquet; su equivalente publicado es `proceso_competencia`.
            ("ofertas enlazadas por id_portafolio (bug corregido)",
             "SELECT CASE WHEN count(*) > 0 THEN 0 ELSE 1 END FROM "
             "proceso_competencia WHERE n_oferentes > 0"),
        ]
        for etiqueta, sql in pruebas:
            try:
                v = c.execute(sql).fetchone()[0]
                print(f"  {'ok      ' if v == 0 else 'FALLA   '} {etiqueta}"
                      + ("" if v == 0 else f"  -> {v:,}"))
                if v != 0:
                    fallos.append(etiqueta)
            except Exception as exc:
                print(f"  ERROR    {etiqueta}: {str(exc)[:50]}")
                fallos.append(etiqueta)

    print("\n[4] Coherencia del universo de red")
    if {"nodo_2025", "vinculo_2025", "contrato_2025"} <= montadas:
        c = con
        c.execute("""CREATE OR REPLACE TEMP VIEW _nodos AS
                     SELECT id_nodo FROM nodo_2025
                     UNION SELECT id_nodo FROM nodo_entidad
                     UNION SELECT id_nodo FROM nodo_representante""")
        pruebas = [
            ("id_nodo unico", "SELECT count(*) - count(DISTINCT id_nodo) FROM nodo_2025"),
            ("sin aristas huerfanas",
             """SELECT count(*) FROM vinculo_2025 v
                WHERE NOT EXISTS (SELECT 1 FROM _nodos n WHERE n.id_nodo = v.origen)
                   OR NOT EXISTS (SELECT 1 FROM _nodos n WHERE n.id_nodo = v.destino)"""),
            ("sin aristas duplicadas",
             "SELECT count(*) - count(DISTINCT (tipo_vinculo, origen, destino)) "
             "FROM vinculo_2025"),
            ("contratos dentro del rango 2025",
             """SELECT count(*) FROM contrato_2025
                WHERE fecha_firma < DATE '2025-01-01' OR fecha_firma > DATE '2025-12-31'"""),
            ("todo contrato tiene nodo",
             """SELECT count(*) FROM contrato_2025 c
                WHERE NOT EXISTS (SELECT 1 FROM nodo_cuenta k
                                  WHERE k.codigo_proveedor = c.codigo_proveedor)"""),
        ]
        for etiqueta, sql in pruebas:
            try:
                v = c.execute(sql).fetchone()[0]
                print(f"  {'ok      ' if v == 0 else 'FALLA   '} {etiqueta}"
                      + ("" if v == 0 else f"  -> {v:,}"))
                if v != 0:
                    fallos.append(etiqueta)
            except Exception as exc:
                print(f"  ERROR    {etiqueta}: {str(exc)[:50]}")
                fallos.append(etiqueta)

        if a.detalle:
            print("\n[5] Ejemplos")
            print("\n  Un contrato con su proceso y su competencia:")
            print(c.execute("""
                SELECT c.id_contrato, c.id_portafolio, p.id_del_proceso,
                       c.tipo_de_contrato, pc.n_oferentes
                FROM contrato_2025 c JOIN proceso_2025 p USING (id_portafolio)
                JOIN proceso_competencia pc USING (id_portafolio)
                WHERE pc.n_oferentes BETWEEN 3 AND 6 LIMIT 3""").df().to_string(index=False))
            print("\n  Nodos por tipo:")
            print(c.execute("""
                SELECT tipo_nodo, count(*) AS nodos,
                       count(*) FILTER (WHERE cruza_rues) AS con_rues
                FROM nodo_2025 GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))
            print("\n  Vinculos por tipo:")
            print(c.execute("""
                SELECT tipo_vinculo, count(*) AS aristas
                FROM vinculo_2025 GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))

    # --- 5. invariantes de las cifras publicadas --------------------------
    #: Las seis discrepancias que la sesión 1 reconcilió a mano vuelven aquí
    #: convertidas en comprobaciones. Cada una fija una cifra que el documento
    #: publica y la recalcula desde `OUTPUTS/gold`, que es la fuente de verdad.
    #: Se consulta el parquet y no las bases `.duckdb` porque estas guardan
    #: vistas con rutas absolutas y solo resuelven en la máquina que las creó.
    print("\n[5] Invariantes de las cifras publicadas")
    T = lambda n: n
    #: `features_nodo.parquet` es la salida de la etapa de grafo y vive en su
    #: tema, no junto al modelo dimensional. Se resuelve con `artefacto` para
    #: que valga tanto el recálculo propio como el publicado.
    features = rutas.artefacto("features_nodo.parquet").replace("\\", "/")

    invariantes = [
        #: F1.b — la suma que no cuadraba: el texto decía 23.051 rescatados y
        #: 200.974 + 23.051 daba 224.025, no 223.987. El valor real es 23.013.
        ("223.987 contratos = 200.974 + 23.013 rescatados", 223_987,
         f"""SELECT count(*) FILTER (WHERE NOT (proveedor_es_persona_natural
                   AND proveedor_en_nucleo_red
                   AND tipo_de_contrato ILIKE '%restaci%servicio%'))
                 + count(*) FILTER (WHERE proveedor_es_persona_natural
                   AND proveedor_en_nucleo_red
                   AND tipo_de_contrato ILIKE '%restaci%servicio%')
             FROM {T('contrato_2025')}"""),
        ("los rescatados del núcleo de red son 23.013", 23_013,
         f"""SELECT count(*) FROM {T('contrato_2025')}
             WHERE proveedor_es_persona_natural AND proveedor_en_nucleo_red
               AND tipo_de_contrato ILIKE '%restaci%servicio%'"""),
        #: F1.a — dos unidades distintas que el texto llamaba igual.
        ("83.374 cuentas de proveedor en el alcance", 83_374,
         f"SELECT count(DISTINCT codigo_proveedor) FROM {T('contrato_2025')}"),
        ("3.872 entidades contratantes", 3_872,
         f"SELECT count(DISTINCT codigo_entidad) FROM {T('contrato_2025')}"),
        #: F1.e — el texto declaraba 272.960 procesos, que no salían de ningún
        #: artefacto. `proceso_competencia` tiene exactamente 217.363 filas.
        ("217.363 procesos con información de competencia", 217_363,
         f"SELECT count(*) FROM {T('proceso_competencia')}"),
        ("146.565 cuentas de plataforma antes de resolver", 146_565,
         f"SELECT count(*) FROM {T('nodo_cuenta')}"),
        ("139.548 actores económicos tras resolver", 139_548,
         f"SELECT count(*) FROM {T('nodo_2025')}"),
        #: El nexo entre el grafo y la matriz de modelado: si estas dos cifras
        #: divergen, hay proveedores del grafo que ningún modelo ha visto. El
        #: filtro de valores atípicos es parte de la definición: cuatro nodos
        #: cuyos únicos contratos son atípicos quedan fuera de `features_nodo`
        #: y por eso 82.988 actores con contrato dan 82.984 filas.
        ("filas de features_nodo = nodos proveedor con contrato no atípico", 0,
         f"""SELECT abs(
               (SELECT count(*) FROM read_parquet('{features}'))
             - (SELECT count(DISTINCT k.id_nodo)
                FROM {T('contrato_2025')} c
                JOIN {T('nodo_cuenta')} k ON k.codigo_proveedor = c.codigo_proveedor
                WHERE NOT c.es_outlier_valor))"""),
        ("features_nodo tiene 82.984 filas", 82_984,
         f"SELECT count(*) FROM read_parquet('{features}')"),
        #: La pseudo-etiqueta es el percentil 90, de modo que la proporción de
        #: positivos tiene que ser el 10 % por construcción.
        ("8.387 proveedores de riesgo alto = percentil 90", 8_387,
         f"""SELECT count(*) FROM read_parquet('{features}')
             WHERE riesgo_alto = 1"""),
    ]
    for etiqueta, esperado, sql in invariantes:
        try:
            v = con.execute(sql).fetchone()[0]
            if v == esperado:
                print(f"  ok       {etiqueta}")
            else:
                print(f"  FALLA    {etiqueta}  -> {v:,} en vez de {esperado:,}")
                fallos.append(f"{etiqueta}: {v:,} en vez de {esperado:,}")
        except Exception as exc:
            print(f"  ERROR    {etiqueta}: {str(exc)[:60]}")
            fallos.append(etiqueta)

    # --- resumen ----------------------------------------------------------
    print("\n" + "=" * 68)
    if not fallos and not avisos:
        print("TODO CORRECTO. El modelo esta completo y es coherente.")
    else:
        if avisos:
            print(f"AVISOS ({len(avisos)}): conteos distintos de lo esperado")
            for x in avisos:
                print(f"  - {x}")
            print("  Normal si has vuelto a descargar los datos de origen.")
        if fallos:
            print(f"\nFALLOS ({len(fallos)}):")
            for x in fallos:
                print(f"  - {x}")
            print("\n  Ejecuta `python orquestador.py --listar` para ver que etapa falta.")
    print("=" * 68)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
