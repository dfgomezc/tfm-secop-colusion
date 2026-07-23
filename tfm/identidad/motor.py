#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motor reutilizable de resolucion de identidad sobre.
registros administrativos, construido encima de Splink 4 + DuckDB.

No depende de ningun pipeline concreto: se dirige con un fichero de
configuracion JSON y funciona sobre cualquier tabla Parquet o CSV que tenga un
identificador de fila, un nombre y, opcionalmente, un documento de identidad.

Lo que aporta sobre usar Splink directamente
--------------------------------------------
1. AUDITORIA DE BLOQUEO PREVIA. Cuenta cuantos pares generaria cada regla
   ANTES de entrenar y aborta las que exploten. En el caso que motivo esto,
   una regla aparentemente inocente -bloquear por los 8 primeros caracteres
   del nombre- generaba 603 millones de pares porque 33.797 registros
   empezaban por "CONSORCI".

2. u EXACTA PARA NIVELES RAROS. Splink estima u muestreando pares al azar. Un
   nivel muy poco frecuente -por ejemplo "los documentos difieren solo en el
   digito de verificacion"- no aparece nunca en la muestra y se queda sin
   entrenar. El motor lo detecta y calcula su u de forma exacta contando los
   pares que cumplen la condicion sobre el total posible.

3. RESTRICCION DURA POR IDENTIFICADOR. Si dos registros tienen identificador
   fuerte valido y distinto, no son la misma entidad, por mucha evidencia que
   acumule el modelo. Esos pares no se fusionan: se publican aparte como
   vinculo de grupo, que suele ser un hallazgo por si mismo.

4. EXCLUSION DE VEHICULOS EFIMEROS. Ciertas figuras -consorcios, uniones
   temporales, UTE, joint ventures- se crean para una operacion concreta y
   comparten contacto con la empresa que las forma. Tratarlas como actores
   persistentes envenena el modelo.

Uso
---
    python -m tfm.identidad.motor --config config_secop.json auditar
    python -m tfm.identidad.motor --config config_secop.json todo

Etapas: auditar | preparar | entrenar | predecir | agrupar | todo

Dependencias: pip install duckdb splink
"""

import argparse
import json
import os
import sys
import time

import duckdb

try:
    from splink import Linker, DuckDBAPI, SettingsCreator
    from splink.blocking_rule_library import CustomRule
    import splink.comparison_library as cl
except ImportError:                                    # pragma: no cover
    print("Falta Splink: pip install splink", file=sys.stderr)
    raise


# ===========================================================================
# Configuracion
# ===========================================================================
PLANTILLA = {
    "estudio": "nombre del estudio",
    "origen": "ruta a *.parquet o *.csv",
    "salida": "carpeta de salida",
    "columnas": {
        "id": "identificador unico de fila",
        "nombre": "razon social o nombre",
        "identificador": "documento, NIT, CIF... (opcional)",
        "tipo": "tipo de entidad (opcional)",
    },
    # Tipos que no son actores persistentes y se excluyen del emparejamiento.
    "tipos_excluidos": [],
    # Encabezados genericos que se quitan del nombre para poder bloquear.
    "prefijos_genericos": [],
    # Columnas categoricas que se comparan por igualdad exacta.
    "columnas_exactas": [],
    "bloqueo": [],
    # Reglas de altisima precision: coincidir aqui es practicamente garantia de
    # ser la misma entidad. Se usan solo para estimar lambda. Si se dejan
    # vacias se toma la primera regla de bloqueo, que casi nunca es lo correcto.
    "reglas_deterministas": [],
    "recall_deterministas": 0.7,
    "umbrales": {"alto": 0.95, "medio": 0.80},
    "max_pares_u": 8000000,
    # Aborta cualquier regla de bloqueo que supere este numero de pares.
    "tope_pares_por_regla": 20000000,
    "digito_verificacion": True,
}


def cargar_config(ruta):
    with open(ruta, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for clave, valor in PLANTILLA.items():
        cfg.setdefault(clave, valor)
    for clave, valor in PLANTILLA["columnas"].items():
        cfg["columnas"].setdefault(clave, None)
    return cfg


# ===========================================================================
# Utilidades
# ===========================================================================
def conectar(cfg, ram, hilos):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{ram}'")
    con.execute("SET preserve_insertion_order=false")
    if hilos:
        con.execute(f"SET threads={hilos}")
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "dd_identidad")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def lector(origen):
    o = origen.replace("\\", "/")
    return (f"read_csv_auto('{o}')" if o.lower().endswith(".csv")
            else f"read_parquet('{o}')")


def volcar(con, tabla, carpeta, nombre=None):
    nombre = nombre or tabla
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{nombre}.parquet").replace("\\", "/")
    cols = con.execute(f"DESCRIBE {tabla}").fetchall()
    proj = ", ".join(
        (f'CAST("{c[0]}" AS VARCHAR) AS "{c[0]}"'
         if "[]" in c[1] or c[1].upper().startswith(("LIST", "STRUCT", "MAP"))
         else f'"{c[0]}"') for c in cols)
    con.execute(f"COPY (SELECT {proj} FROM {tabla}) TO '{ruta}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD)")
    return ruta


def tabla_entrada(cfg):
    return os.path.join(cfg["salida"], "entrada.parquet")


# ===========================================================================
# Etapa 0 — preparar
# ===========================================================================
def etapa_preparar(cfg, con):
    c = cfg["columnas"]
    os.makedirs(cfg["salida"], exist_ok=True)

    prefijos = "|".join(cfg["prefijos_genericos"])
    nucleo = (f"regexp_replace(_nombre, '^({prefijos})\\s+', '')"
              if prefijos else "_nombre")

    extra = ", ".join(f'"{x}"' for x in cfg["columnas_exactas"])
    extra = (", " + extra) if extra else ""

    ident = f'"{c["identificador"]}"' if c["identificador"] else "NULL"
    tipo = f'"{c["tipo"]}"' if c["tipo"] else "'sin_tipo'"

    con.execute(f"""
        CREATE OR REPLACE TABLE entrada AS
        WITH base AS (
            SELECT
                CAST("{c['id']}" AS VARCHAR)              AS unique_id,
                nullif(trim(regexp_replace(upper(strip_accents(
                    CAST("{c['nombre']}" AS VARCHAR))), '\\s+', ' ', 'g')), '') AS _nombre,
                CAST({ident} AS VARCHAR)                  AS identificador,
                CAST({tipo} AS VARCHAR)                   AS tipo_entidad
                {extra}
            FROM {lector(cfg['origen'])}
        )
        SELECT
            unique_id,
            _nombre                                        AS nombre,
            regexp_replace(_nombre, '[^A-Z0-9]', '', 'g')  AS nombre_compacto,
            nullif(trim({nucleo}), '')                     AS nombre_nucleo,
            identificador,
            tipo_entidad
            {extra}
        FROM base
        WHERE _nombre IS NOT NULL
    """)
    ruta = volcar(con, "entrada", cfg["salida"])
    n = con.execute("SELECT count(*) FROM entrada").fetchone()[0]
    print(f"  entrada: {n:,} registros -> {ruta}")
    print(con.execute("""
        SELECT tipo_entidad, count(*) AS registros,
               count(identificador) AS con_identificador
        FROM entrada GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").df().to_string(index=False))
    return n


def cargar_entrada(cfg, con):
    ruta = tabla_entrada(cfg)
    if not os.path.exists(ruta):
        raise SystemExit("ejecuta antes la etapa 'preparar'")
    excl = cfg["tipos_excluidos"]
    filtro = ""
    if excl:
        lista = ", ".join(f"'{t}'" for t in excl)
        filtro = f"WHERE tipo_entidad NOT IN ({lista})"
    con.execute(f"CREATE OR REPLACE TABLE entrada AS "
                f"SELECT * FROM {lector(ruta)} {filtro}")
    return con.execute("SELECT count(*) FROM entrada").fetchone()[0]


# ===========================================================================
# Etapa 1 — auditar el bloqueo
# ===========================================================================
def etapa_auditar(cfg, con):
    """Cuenta los pares que generaria cada regla ANTES de entrenar."""
    n = cargar_entrada(cfg, con)
    total = n * (n - 1) / 2
    print(f"  registros tras excluir {cfg['tipos_excluidos'] or 'nada'}: {n:,}")
    print(f"  pares posibles sin bloqueo: {total:,.0f}\n")

    filas, veredicto_global = [], True
    for regla in cfg["bloqueo"]:
        expr = (regla if isinstance(regla, str) else regla[0]).format(x="e")
        try:
            pares = con.execute(f"""
                SELECT coalesce(sum(n * (n - 1) / 2), 0)
                FROM (SELECT {expr} AS k, count(*) AS n
                      FROM entrada e WHERE {expr} IS NOT NULL GROUP BY 1)
            """).fetchone()[0]
        except Exception as exc:
            print(f"  [ERROR] regla invalida: {regla}\n          {exc}")
            veredicto_global = False
            continue
        ok = pares <= cfg["tope_pares_por_regla"]
        veredicto_global &= ok
        filas.append((regla, int(pares), 100 * pares / total, "ok" if ok else "EXCESIVA"))

    ancho = max(len(f[0]) for f in filas) if filas else 10
    print(f"  {'regla':<{ancho}}  {'pares':>16}  {'% del total':>11}  veredicto")
    for r, p, pct, v in sorted(filas, key=lambda x: -x[1]):
        print(f"  {r:<{ancho}}  {p:>16,}  {pct:>10.4f}%  {v}")

    if not veredicto_global:
        print(f"\n  [!] Hay reglas por encima del tope "
              f"({cfg['tope_pares_por_regla']:,} pares).")
        print("      Suele indicar un valor muy repetido: revisa el ranking de")
        print("      frecuencias mas abajo y usa 'nombre_nucleo' o añade el")
        print("      prefijo a 'prefijos_genericos'.")
        print(con.execute("""
            SELECT substr(nombre, 1, 8) AS prefijo, count(*) AS registros
            FROM entrada GROUP BY 1 ORDER BY 2 DESC LIMIT 8""").df().to_string(index=False))
    return veredicto_global


# ===========================================================================
# u exacta para niveles raros
# ===========================================================================
def u_digito_verificacion(con):
    """P(los documentos difieren solo en el ultimo caracter | no son el mismo).

    Se calcula de forma exacta porque el muestreo aleatorio de Splink nunca
    llega a observar este nivel: es demasiado infrecuente."""
    n = con.execute("SELECT count(*) FROM entrada").fetchone()[0]
    total = n * (n - 1) / 2
    casos = con.execute("""
        SELECT count(*) FROM entrada a JOIN entrada b
          ON b.identificador = substr(a.identificador, 1,
                                      length(a.identificador) - 1)
        WHERE a.identificador IS NOT NULL AND b.identificador IS NOT NULL
          AND a.unique_id <> b.unique_id
    """).fetchone()[0]
    u = casos / total if total else 0.0
    print(f"  u exacta del nivel 'digito de verificacion': {casos:,} pares "
          f"sobre {total:,.0f} -> {u:.3e}")
    return max(u, 1e-12), casos


def comparacion_identificador(u_dv):
    return {
        "output_column_name": "identificador",
        "comparison_description": "Identificador, con nivel propio para el digito de control",
        "comparison_levels": [
            {"sql_condition": '"identificador_l" IS NULL OR "identificador_r" IS NULL',
             "label_for_charts": "Alguno nulo", "is_null_level": True},
            {"sql_condition": '"identificador_l" = "identificador_r"',
             "label_for_charts": "Identico"},
            {"sql_condition":
                '"identificador_l" = substr("identificador_r", 1, length("identificador_r") - 1) '
                'OR "identificador_r" = substr("identificador_l", 1, length("identificador_l") - 1)',
             "label_for_charts": "Difiere solo en el digito de control",
             "u_probability": u_dv},
            {"sql_condition": 'levenshtein("identificador_l", "identificador_r") <= 1',
             "label_for_charts": "Difiere en un caracter"},
            {"sql_condition": 'levenshtein("identificador_l", "identificador_r") <= 2',
             "label_for_charts": "Difiere en dos caracteres"},
            {"sql_condition": "ELSE", "label_for_charts": "Distinto"},
        ],
    }


def construir_settings(cfg, u_dv):
    reglas = [CustomRule(f"{r.format(x='l')} = {r.format(x='r')} "
                         f"AND l.tipo_entidad = r.tipo_entidad")
              for r in cfg["bloqueo"]]
    comparaciones = [
        cl.JaroWinklerAtThresholds("nombre", [0.95, 0.88, 0.80])
          .configure(term_frequency_adjustments=True)
    ]
    if cfg["columnas"]["identificador"]:
        comparaciones.append(comparacion_identificador(u_dv))
    for col in cfg["columnas_exactas"]:
        comparaciones.append(cl.ExactMatch(col))
    return SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=reglas,
        comparisons=comparaciones,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


# ===========================================================================
# Etapa 2 — entrenar
# ===========================================================================
def etapa_entrenar(cfg, con, ram, hilos):
    cargar_entrada(cfg, con)
    u_dv, _ = (u_digito_verificacion(con)
               if cfg["digito_verificacion"] and cfg["columnas"]["identificador"]
               else (None, 0))
    lk = Linker("entrada", construir_settings(cfg, u_dv), db_api=DuckDBAPI(connection=con))

    def a_sql(plantilla):
        """Una plantilla puede ser una cadena o una lista de cadenas que se
        combinan con AND. No se usa concat() para componer claves: en DuckDB
        concat trata NULL como cadena vacia, asi que todos los registros con
        el campo vacio casarian entre si."""
        partes = plantilla if isinstance(plantilla, list) else [plantilla]
        return " AND ".join(f"{p.format(x='l')} = {p.format(x='r')}"
                            for p in partes)

    plantillas = cfg["reglas_deterministas"] or cfg["bloqueo"][:1]
    deterministas = [CustomRule(a_sql(p)) for p in plantillas]
    lk.training.estimate_probability_two_random_records_match(
        deterministas, recall=cfg["recall_deterministas"])
    lk.training.estimate_u_using_random_sampling(max_pairs=int(cfg["max_pares_u"]))

    for regla in cfg["bloqueo"]:
        try:
            lk.training.estimate_parameters_using_expectation_maximisation(
                CustomRule(f"{regla.format(x='l')} = {regla.format(x='r')}"))
        except Exception as exc:
            print(f"  [!] EM con bloqueo {regla}: {str(exc)[:80]}")

    ruta = os.path.join(cfg["salida"], "modelo.json")
    lk.misc.save_model_to_json(ruta, overwrite=True)

    with open(ruta, encoding="utf-8") as fh:
        modelo = json.load(fh)
    filas = [(c.get("output_column_name"), n.get("label_for_charts"),
              n.get("m_probability"), n.get("u_probability"))
             for c in modelo["comparisons"] for n in c["comparison_levels"]]
    con.execute("CREATE OR REPLACE TABLE parametros AS SELECT * FROM (VALUES " +
                ", ".join("(?, ?, ?, ?)" for _ in filas) +
                ") AS t(comparacion, nivel, m, u)",
                [x for f in filas for x in f])
    volcar(con, "parametros", cfg["salida"])
    print(f"\n  lambda = {modelo['probability_two_random_records_match']:.3e}")
    print(con.execute("""
        SELECT comparacion, nivel, round(m, 5) AS m, round(u, 9) AS u,
               round(log2(m / nullif(u, 0)), 2) AS peso_bits
        FROM parametros WHERE m IS NOT NULL
        ORDER BY peso_bits DESC NULLS LAST""").df().to_string(index=False))
    print(f"\n  modelo -> {ruta}")


# ===========================================================================
# Etapa 3 — predecir
# ===========================================================================
def etapa_predecir(cfg, con, ram, hilos):
    cargar_entrada(cfg, con)
    with open(os.path.join(cfg["salida"], "modelo.json"), encoding="utf-8") as fh:
        ajustes = json.load(fh)
    lk = Linker("entrada", ajustes, db_api=DuckDBAPI(connection=con))
    pred = lk.inference.predict(
        threshold_match_probability=cfg["umbrales"]["medio"])
    con.execute(f"CREATE OR REPLACE TABLE pares AS "
                f"SELECT * FROM {pred.physical_name}")
    volcar(con, "pares", cfg["salida"])
    alto = cfg["umbrales"]["alto"]
    print(con.execute(f"""
        SELECT CASE WHEN match_probability >= {alto} THEN '1. >= {alto}'
                    ELSE '2. zona de revision' END AS tramo,
               count(*) AS pares,
               round(min(match_weight), 2) AS peso_min,
               round(max(match_weight), 2) AS peso_max
        FROM pares GROUP BY 1 ORDER BY 1""").df().to_string(index=False))


# ===========================================================================
# Etapa 4 — agrupar, con la restriccion dura por identificador
# ===========================================================================
def etapa_agrupar(cfg, con, ram, hilos):
    cargar_entrada(cfg, con)
    ruta = os.path.join(cfg["salida"], "pares.parquet")
    if not os.path.exists(ruta):
        raise SystemExit("ejecuta antes la etapa 'predecir'")
    con.execute(f"CREATE OR REPLACE TABLE pares AS SELECT * FROM {lector(ruta)}")
    alto = cfg["umbrales"]["alto"]

    # ------------------------------------------------------------------
    # Restriccion dura: dos identificadores validos y distintos implican dos
    # entidades distintas, por mucha evidencia que acumule el modelo. Esos
    # pares no se fusionan; se publican como vinculo de grupo.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE clasificado AS
        SELECT p.*, l.identificador AS id_l, rr.identificador AS id_r,
               CASE
                 WHEN l.identificador IS NULL OR rr.identificador IS NULL
                      THEN 'fusionable'
                 WHEN l.identificador = rr.identificador THEN 'fusionable'
                 WHEN l.identificador = substr(rr.identificador, 1,
                                               length(rr.identificador) - 1)
                   OR rr.identificador = substr(l.identificador, 1,
                                                length(l.identificador) - 1)
                      THEN 'fusionable'
                 ELSE 'mismo_grupo' END AS veredicto
        FROM pares p
        JOIN entrada l  ON l.unique_id  = p.unique_id_l
        JOIN entrada rr ON rr.unique_id = p.unique_id_r
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE grupo AS
        SELECT unique_id_l AS id_a, unique_id_r AS id_b,
               round(match_probability, 4) AS probabilidad,
               round(match_weight, 2) AS peso_bits, id_l, id_r
        FROM clasificado
        WHERE veredicto = 'mismo_grupo' AND match_probability >= {alto}
    """)
    volcar(con, "grupo", cfg["salida"], "vinculo_mismo_grupo")

    # Componentes conexas sobre los pares fusionables
    con.execute(f"""
        CREATE OR REPLACE TABLE arista AS
        SELECT unique_id_l AS a, unique_id_r AS b FROM clasificado
        WHERE match_probability >= {alto} AND veredicto = 'fusionable'
    """)
    con.execute("""
        CREATE OR REPLACE TABLE comp AS
        SELECT id, id AS lider FROM (
            SELECT a AS id FROM arista UNION SELECT b FROM arista)
    """)
    for _ in range(30):
        cambios = con.execute("""
            CREATE OR REPLACE TABLE comp_nuevo AS
            WITH bi AS (SELECT a AS x, b AS y FROM arista
                        UNION ALL SELECT b, a FROM arista)
            SELECT c.id, least(c.lider, coalesce(min(c2.lider), c.lider)) AS lider
            FROM comp c
            LEFT JOIN bi ON bi.x = c.id
            LEFT JOIN comp c2 ON c2.id = bi.y
            GROUP BY c.id, c.lider;
            SELECT count(*) FROM comp c JOIN comp_nuevo n USING (id)
             WHERE c.lider <> n.lider
        """).fetchone()[0]
        con.execute("DROP TABLE comp; ALTER TABLE comp_nuevo RENAME TO comp")
        if cambios == 0:
            break

    con.execute("""
        CREATE OR REPLACE TABLE cluster AS
        SELECT e.unique_id AS id, coalesce(c.lider, e.unique_id) AS id_resuelto,
               c.lider IS NOT NULL AS fusionado
        FROM entrada e LEFT JOIN comp c ON c.id = e.unique_id
    """)
    volcar(con, "cluster", cfg["salida"])

    con.execute(f"""
        CREATE OR REPLACE TABLE evidencia AS
        SELECT
            CASE WHEN c.match_probability < {alto} THEN 'revisar'
                 WHEN c.veredicto = 'mismo_grupo' THEN 'mismo grupo'
                 ELSE 'fusionar' END              AS decision,
            round(c.match_probability, 4)         AS probabilidad,
            round(c.match_weight, 2)              AS peso_bits,
            c.unique_id_l, c.unique_id_r,
            l.nombre AS nombre_a, rr.nombre AS nombre_b,
            c.id_l, c.id_r,
            CASE WHEN c.id_l = c.id_r THEN 'identico'
                 WHEN c.id_l IS NULL OR c.id_r IS NULL THEN 'sin identificador'
                 WHEN c.id_l = substr(c.id_r, 1, length(c.id_r) - 1)
                   OR c.id_r = substr(c.id_l, 1, length(c.id_l) - 1)
                      THEN 'digito de control'
                 ELSE 'distinto' END              AS relacion_identificador
        FROM clasificado c
        JOIN entrada l  ON l.unique_id  = c.unique_id_l
        JOIN entrada rr ON rr.unique_id = c.unique_id_r
    """)
    volcar(con, "evidencia", cfg["salida"])

    print(con.execute("""
        SELECT decision, count(*) AS pares,
               count(*) FILTER (WHERE relacion_identificador = 'digito de control') AS por_digito,
               count(*) FILTER (WHERE relacion_identificador = 'sin identificador') AS sin_id
        FROM evidencia GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    print()
    print(con.execute("""
        SELECT count(*) FILTER (WHERE fusionado) AS registros_fusionados,
               count(DISTINCT id_resuelto) FILTER (WHERE fusionado) AS grupos,
               count(*) AS registros, count(DISTINCT id_resuelto) AS entidades
        FROM cluster""").df().to_string(index=False))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("etapa", choices=["auditar", "preparar", "entrenar",
                                      "predecir", "agrupar", "todo"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    a = ap.parse_args()

    cfg = cargar_config(a.config)
    os.makedirs(cfg["salida"], exist_ok=True)
    print(f"=== {cfg['estudio']} ===")

    etapas = (["preparar", "auditar", "entrenar", "predecir", "agrupar"]
              if a.etapa == "todo" else [a.etapa])
    for e in etapas:
        print(f"\n[{e}]")
        t0 = time.time()
        con = conectar(cfg, a.ram, a.hilos)
        if e == "preparar":
            etapa_preparar(cfg, con)
        elif e == "auditar":
            if not etapa_auditar(cfg, con) and a.etapa == "todo":
                print("\n[!] Auditoria fallida: corrige el bloqueo antes de seguir.")
                return
        elif e == "entrenar":
            etapa_entrenar(cfg, con, a.ram, a.hilos)
        elif e == "predecir":
            etapa_predecir(cfg, con, a.ram, a.hilos)
        elif e == "agrupar":
            etapa_agrupar(cfg, con, a.ram, a.hilos)
        con.close()
        print(f"  ({time.time() - t0:.1f}s)")

    print(f"\nSalidas en {cfg['salida']}")


if __name__ == "__main__":
    main()
