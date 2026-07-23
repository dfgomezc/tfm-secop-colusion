#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolucion probabilistica de identidad de nodos con Splink 4.

Complementa a la resolucion determinista de `nodos.sql`. Aquella solo une lo
que se puede demostrar (mismo documento base, con el RUES arbitrando el digito
de verificacion). Esta estima la probabilidad de que dos nodos sean el mismo
actor a partir de la coincidencia conjunta de razon social, documento,
representante legal, contacto, camara y territorio.

Se ejecuta por etapas, para poder inspeccionar cada paso:

    python -m tfm.identidad.splink_resolver preparar     # construye la tabla de entrada
    python -m tfm.identidad.splink_resolver entrenar     # estima u por muestreo y m por EM
    python -m tfm.identidad.splink_resolver predecir     # calcula los pares y su probabilidad
    python -m tfm.identidad.splink_resolver agrupar      # clusters y tablas de evidencia
    python -m tfm.identidad.splink_resolver todo         # las cuatro seguidas

Salidas en OUTPUTS/gold/ y OUTPUTS/gold/splink/:
    splink_entrada       un registro por nodo, con los atributos comparados
    splink_parametros    m y u aprendidas, por comparacion y nivel
    splink_pares         pares candidatos con match_weight y match_probability
    splink_cluster       id_nodo -> id_nodo_resuelto
    splink_evidencia     pares lado a lado, por tramo de probabilidad
    modelo.json          el modelo entrenado, reutilizable
"""

import argparse
import glob
import json
import os

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()

# Umbrales: por encima de ALTO se propone fusionar; entre MEDIO y ALTO queda
# para revision manual. Se pueden cambiar sin reentrenar.
UMBRAL_ALTO = 0.95
UMBRAL_MEDIO = 0.80


# ---------------------------------------------------------------------------
# Comparacion a medida del documento.
#
# El caso que interesa: en Colombia el NIT de una persona natural es su cedula
# mas un digito de verificacion, y el mismo actor aparece unas veces con uno y
# otras con otro. La resolucion determinista solo recorta el DV cuando el RUES
# lo confirma; aqui ese parecido entra como nivel propio y Splink aprende
# cuanto pesa a partir de los datos.
# ---------------------------------------------------------------------------
COMPARACION_DOCUMENTO = {
    "output_column_name": "documento",
    "comparison_description": "Documento con nivel propio para el digito de verificacion",
    "comparison_levels": [
        {"sql_condition": '"documento_l" IS NULL OR "documento_r" IS NULL',
         "label_for_charts": "Alguno nulo", "is_null_level": True},
        # No se incluye un nivel "documento identico": es estructuralmente
        # imposible. La resolucion determinista de `nodos.sql` ya agrupo por
        # documento base, asi que dos nodos distintos nunca comparten documento.
        # u fijada a mano: el muestreo aleatorio nunca llega a ver este nivel
        # (8 M de pares muestreados y ninguno cae aqui). Se calcula de forma
        # exacta: 464 pares con esta relacion sobre 9.502.896.591 posibles.
        # Nota: la relacion por si sola NO implica identidad. Entre esos 464
        # hay casos como UT NAVEGACION / CONSORCIO GEOWILCHES, con documentos
        # basura del tipo 0001112223. Tiene que acompañarla el nombre.
        {"sql_condition":
            '"documento_l" = substr("documento_r", 1, length("documento_r") - 1) '
            'OR "documento_r" = substr("documento_l", 1, length("documento_l") - 1)',
         "label_for_charts": "Difiere solo en el digito de verificacion",
         "u_probability": 4.883e-08},
        {"sql_condition": 'levenshtein("documento_l", "documento_r") <= 1',
         "label_for_charts": "Difiere en un caracter"},
        {"sql_condition": 'levenshtein("documento_l", "documento_r") <= 2',
         "label_for_charts": "Difiere en dos caracteres"},
        {"sql_condition": "ELSE", "label_for_charts": "Distinto"},
    ],
}


def rutas(raiz):
    gold = os.path.join(raiz, "gold")
    return {
        "gold": gold,
        "pq": os.path.join(gold, "parquet"),
        "sp": os.path.join(gold, "splink"),
        "bd": os.path.join(gold, "secop_tfm_red2025.duckdb"),
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
    return sorted(f for f in glob.glob(os.path.join(carpeta, "*.parquet"))
                  if parquet_valido(f))


def lista_sql(fs):
    return "[" + ", ".join("'" + f.replace("\\", "/").replace("'", "''") + "'"
                           for f in fs) + "]"


def publicar(con, tabla, pq, nombre=None):
    nombre = nombre or tabla
    destino = os.path.join(pq, nombre)
    os.makedirs(destino, exist_ok=True)
    for viejo in glob.glob(os.path.join(destino, "*.parquet")):
        os.remove(viejo)
    cols = con.execute(f"DESCRIBE {tabla}").fetchall()
    proj = ", ".join(
        (f'CAST("{c[0]}" AS VARCHAR) AS "{c[0]}"'
         if "[]" in c[1] or c[1].upper().startswith(("LIST", "STRUCT", "MAP"))
         else f'"{c[0]}"') for c in cols)
    ruta = os.path.join(destino, "part_0000.parquet").replace("\\", "/")
    con.execute(f"COPY (SELECT {proj} FROM {tabla}) TO '{ruta}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD)")
    return ruta


# ---------------------------------------------------------------------------
def etapa_preparar(r, ram, hilos):
    """Construye splink_entrada: un registro por nodo con lo que se compara."""
    con = duckdb.connect(r["bd"], read_only=True)
    con.execute(f"SET memory_limit='{ram}'")
    if hilos:
        con.execute(f"SET threads={hilos}")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE splink_entrada AS
        SELECT
            n.id_nodo                                   AS unique_id,
            n.nombre_canonico,
            -- nombre sin espacios: absorbe separaciones distintas del mismo nombre
            regexp_replace(n.nombre_canonico, '[^A-Z0-9]', '', 'g') AS nombre_compacto,
            -- nucleo del nombre: se quita el encabezado generico. 33.797 nodos
            -- empiezan por 'CONSORCI', asi que bloquear por el prefijo crudo
            -- generaria 603 millones de pares. Quitandolo bajan a unos miles.
            nullif(trim(regexp_replace(n.nombre_canonico,
                '^(CONSORCIO|UNION TEMPORAL|UT|U T|CONSORCIA|JUNTA DE ACCION COMUNAL|'
                'JUNTA DE|FUNDACION|ASOCIACION|CORPORACION|COOPERATIVA|'
                'INSTITUCION EDUCATIVA|EMPRESA|SOCIEDAD)\s+', '')), '') AS nombre_nucleo,
            n.documento,
            n.tipo_nodo,
            n.departamento_norm,
            n.municipio,
            n.camara_comercio,
            n.ciiu1,
            n.representante_legal_canonico,
            n.doc_representante_legal,
            lower(n.correo)                             AS correo,
            regexp_replace(n.telefono, '[^0-9]', '', 'g') AS telefono,
            n.estado_matricula,
            n.n_contratos_2025,
            n.n_procesos_2025,
            n.fue_adjudicado,
            n.presento_oferta
        FROM nodo_2025 n
        WHERE n.nombre_canonico IS NOT NULL
    """)
    os.makedirs(r["sp"], exist_ok=True)
    ruta = publicar(con, "splink_entrada", r["pq"])
    n = con.execute("SELECT count(*) FROM splink_entrada").fetchone()[0]
    print(f"  splink_entrada: {n:,} nodos -> {ruta}")
    print(con.execute("""SELECT tipo_nodo, count(*) AS nodos,
                                count(documento) AS con_documento,
                                count(representante_legal_canonico) AS con_rep_legal,
                                count(correo) AS con_correo
                         FROM splink_entrada GROUP BY 1 ORDER BY 2 DESC""")
          .df().to_string(index=False))
    con.close()


def _settings(block_on, cl):
    """Configuracion del modelo. Las reglas de bloqueo definen que pares se
    llegan a comparar: sin ellas habria 9.502.896.591 combinaciones.

    Todas las reglas exigen ademas que los dos nodos sean del mismo tipo. Sin
    esa condicion, el modelo se llenaba de pares union temporal / empresa
    miembro que comparten correo y telefono (84.619 pares, del tipo
    CONSORCIO JHS PROCESOS 2023 con JHS COLOMBIANA DE INGENIERIA). Compartir
    contacto ahi prueba una relacion, no una identidad: esa señal ya esta
    recogida como arista `comparte_contacto` e `integra` en vinculo_2025.
    """
    from splink.blocking_rule_library import CustomRule

    def blk(plantilla):
        """La plantilla usa {x} donde va el alias de cada lado."""
        izq, der = plantilla.format(x="l"), plantilla.format(x="r")
        return CustomRule(f"{izq} = {der} AND l.tipo_nodo = r.tipo_nodo")

    return dict(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            blk("{x}.nombre_compacto"),
            blk("substr({x}.nombre_nucleo, 1, 10)"),
            blk("{x}.doc_representante_legal"),
            blk("{x}.correo"),
            blk("{x}.telefono"),
            # captura directamente los pares que difieren en el digito de
            # verificacion, que ninguna otra regla llegaria a proponer
            blk("substr({x}.documento, 1, length({x}.documento) - 1)"),
        ],
        comparisons=[
            cl.JaroWinklerAtThresholds("nombre_canonico", [0.95, 0.88, 0.80])
              .configure(term_frequency_adjustments=True),
            COMPARACION_DOCUMENTO,
            cl.ExactMatch("doc_representante_legal")
              .configure(term_frequency_adjustments=True),
            cl.ExactMatch("correo"),
            cl.ExactMatch("telefono"),
            cl.ExactMatch("departamento_norm"),
            cl.ExactMatch("camara_comercio"),
            cl.ExactMatch("ciiu1"),
        ],
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _linker(r, ram, hilos, cargar_modelo=False):
    from splink import Linker, DuckDBAPI, SettingsCreator, block_on
    import splink.comparison_library as cl

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{ram}'")
    con.execute("SET preserve_insertion_order=false")
    if hilos:
        con.execute(f"SET threads={hilos}")
    fs = parquets(os.path.join(r["pq"], "splink_entrada"))
    if not fs:
        raise SystemExit("ejecuta antes: 07_splink.py preparar")
    # Las uniones temporales quedan fuera de la resolucion de identidad: un
    # consorcio es un vehiculo que se crea para una licitacion concreta, no un
    # actor persistente, y sus datos de contacto son los de la empresa lider.
    # Dos consorcios con el mismo correo no son el mismo consorcio.
    con.execute(f"CREATE OR REPLACE TABLE entrada AS SELECT * FROM "
                f"read_parquet({lista_sql(fs)}) "
                f"WHERE tipo_nodo <> 'union_temporal'")

    db_api = DuckDBAPI(connection=con)
    if cargar_modelo:
        with open(os.path.join(r["sp"], "modelo.json"), encoding="utf-8") as fh:
            ajustes = json.load(fh)
        return Linker("entrada", ajustes, db_api=db_api), con, block_on
    return (Linker("entrada", SettingsCreator(**_settings(block_on, cl)),
                   db_api=db_api), con, block_on)


def etapa_entrenar(r, ram, hilos, max_pares):
    lk, con, block_on = _linker(r, ram, hilos)

    # 1. Cuantos pares al azar son de verdad la misma entidad. Se estima con
    #    reglas deterministas: coincidencias que casi con certeza son un match.
    #    Ojo: no sirve bloquear por documento, porque tras la resolucion
    #    determinista dos nodos nunca comparten documento (daria cero pares).
    lk.training.estimate_probability_two_random_records_match(
        [block_on("nombre_compacto"),
         block_on("doc_representante_legal", "departamento_norm")],
        recall=0.7)

    # 2. u = probabilidad de coincidir por azar entre no-matches. Se estima
    #    muestreando pares al azar: casi todos son no-matches.
    lk.training.estimate_u_using_random_sampling(max_pairs=max_pares)

    # 3. m = probabilidad de coincidir entre matches reales, por EM. Cada
    #    pasada bloquea por una columna, asi que hacen falta varias para
    #    entrenar todas: la columna bloqueada no se entrena en esa pasada.
    for regla in ("doc_representante_legal", "nombre_compacto", "correo"):
        try:
            lk.training.estimate_parameters_using_expectation_maximisation(
                block_on(regla))
        except Exception as e:                      # pasada sin pares suficientes
            print(f"  [!] EM con bloqueo {regla}: {str(e)[:90]}")

    os.makedirs(r["sp"], exist_ok=True)
    ruta_modelo = os.path.join(r["sp"], "modelo.json")
    lk.misc.save_model_to_json(ruta_modelo, overwrite=True)

    # Parametros aprendidos, en formato legible
    with open(ruta_modelo, encoding="utf-8") as fh:
        modelo = json.load(fh)
    filas = []
    for c in modelo["comparisons"]:
        for niv in c["comparison_levels"]:
            filas.append((c.get("output_column_name"),
                          niv.get("label_for_charts"),
                          niv.get("m_probability"),
                          niv.get("u_probability")))
    con.execute("CREATE OR REPLACE TABLE splink_parametros AS "
                "SELECT * FROM (VALUES " +
                ", ".join("(?, ?, ?, ?)" for _ in filas) +
                ") AS t(comparacion, nivel, m_probability, u_probability)",
                [x for f in filas for x in f])
    publicar(con, "splink_parametros", r["pq"])
    print(con.execute("""SELECT comparacion, nivel,
                                round(m_probability, 5) AS m,
                                round(u_probability, 7) AS u,
                                round(log2(m_probability / u_probability), 2) AS peso_bits
                         FROM splink_parametros
                         WHERE m_probability IS NOT NULL""").df().to_string(index=False))
    con.close()
    print(f"\n  modelo -> {os.path.join(r['sp'], 'modelo.json')}")


def etapa_predecir(r, ram, hilos):
    lk, con, _ = _linker(r, ram, hilos, cargar_modelo=True)
    pred = lk.inference.predict(threshold_match_probability=UMBRAL_MEDIO)
    con.execute(f"CREATE OR REPLACE TABLE splink_pares AS "
                f"SELECT * FROM {pred.physical_name}")
    publicar(con, "splink_pares", r["pq"])
    print(con.execute(f"""
        SELECT CASE WHEN match_probability >= {UMBRAL_ALTO} THEN '1. >= {UMBRAL_ALTO}'
                    WHEN match_probability >= 0.9  THEN '2. 0.90 - {UMBRAL_ALTO}'
                    ELSE '3. {UMBRAL_MEDIO} - 0.90' END AS tramo,
               count(*) AS pares,
               round(min(match_weight), 2) AS peso_min,
               round(max(match_weight), 2) AS peso_max
        FROM splink_pares GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    con.close()


def etapa_agrupar(r, ram, hilos):
    lk, con, _ = _linker(r, ram, hilos, cargar_modelo=True)
    fs = parquets(os.path.join(r["pq"], "splink_pares"))
    if not fs:
        raise SystemExit("ejecuta antes: 07_splink.py predecir")
    con.execute(f"CREATE OR REPLACE TABLE splink_pares AS "
                f"SELECT * FROM read_parquet({lista_sql(fs)})")
    fn = parquets(os.path.join(r["pq"], "nodo_2025"))
    con.execute(f"CREATE OR REPLACE VIEW nodo_2025 AS SELECT * FROM "
                f"read_parquet({lista_sql(fn)}, union_by_name=true)")

    # ------------------------------------------------------------------
    # Regla de dominio que Splink no puede conocer: si los dos nodos tienen
    # documento valido y distinto (y no es la relacion de digito de
    # verificacion), son personas juridicas DISTINTAS, por mucho que compartan
    # nombre, representante legal y telefono. El caso tipico es
    # SETEC GOMEZ CAJIAO (860034335) con SETEC ANDINA (901399529), o
    # SONDA DE COLOMBIA con SONDA S.A.: mismo grupo empresarial, NIT distinto.
    #
    # Esos pares NO se fusionan; se publican como vinculo de grupo empresarial,
    # que para el analisis de colusion es tanto o mas interesante que la
    # identidad, porque son oferentes formalmente independientes.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE par_clasificado AS
        SELECT p.*,
               l.documento AS doc_l, rgt.documento AS doc_r,
               CASE
                 WHEN l.documento IS NULL OR rgt.documento IS NULL THEN 'fusionable'
                 WHEN l.documento = substr(rgt.documento, 1, length(rgt.documento) - 1)
                   OR rgt.documento = substr(l.documento, 1, length(l.documento) - 1)
                      THEN 'fusionable'
                 WHEN l.documento = rgt.documento THEN 'fusionable'
                 ELSE 'grupo_empresarial' END AS veredicto
        FROM splink_pares p
        JOIN nodo_2025 l   ON l.id_nodo   = p.unique_id_l
        JOIN nodo_2025 rgt ON rgt.id_nodo = p.unique_id_r
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE splink_grupo_empresarial AS
        SELECT unique_id_l AS id_nodo_a, unique_id_r AS id_nodo_b,
               round(match_probability, 4) AS probabilidad,
               round(match_weight, 2)      AS peso_bits,
               doc_l, doc_r
        FROM par_clasificado
        WHERE veredicto = 'grupo_empresarial'
          AND match_probability >= {UMBRAL_ALTO}
    """)
    publicar(con, "splink_grupo_empresarial", r["pq"])

    # Componentes conexas por encima del umbral alto, solo con pares fusionables
    con.execute(f"""
        CREATE OR REPLACE TABLE arista AS
        SELECT unique_id_l AS a, unique_id_r AS b
        FROM par_clasificado
        WHERE match_probability >= {UMBRAL_ALTO} AND veredicto = 'fusionable'
    """)
    con.execute("""
        CREATE OR REPLACE TABLE nodo_lista AS
        SELECT a AS id FROM arista UNION SELECT b FROM arista
    """)
    # union-find iterativo: se propaga el minimo id por la componente
    con.execute("""
        CREATE OR REPLACE TABLE comp AS
        SELECT id, id AS lider FROM nodo_lista
    """)
    for _ in range(25):
        cambios = con.execute("""
            CREATE OR REPLACE TABLE comp_nuevo AS
            WITH bi AS (SELECT a AS x, b AS y FROM arista
                        UNION ALL SELECT b, a FROM arista),
                 prop AS (
                    SELECT c.id, least(c.lider, coalesce(min(c2.lider), c.lider)) AS lider
                    FROM comp c
                    LEFT JOIN bi ON bi.x = c.id
                    LEFT JOIN comp c2 ON c2.id = bi.y
                    GROUP BY c.id, c.lider)
            SELECT * FROM prop;
            SELECT count(*) FROM comp c JOIN comp_nuevo n USING (id)
             WHERE c.lider <> n.lider
        """).fetchone()[0]
        con.execute("DROP TABLE comp; ALTER TABLE comp_nuevo RENAME TO comp")
        if cambios == 0:
            break

    con.execute("""
        CREATE OR REPLACE TABLE splink_cluster AS
        SELECT e.unique_id                       AS id_nodo,
               coalesce(c.lider, e.unique_id)    AS id_nodo_resuelto,
               c.lider IS NOT NULL               AS fusionado
        FROM entrada e LEFT JOIN comp c ON c.id = e.unique_id
    """)
    publicar(con, "splink_cluster", r["pq"])

    # Evidencia: los dos registros lado a lado, con lo que los une
    con.execute(f"""
        CREATE OR REPLACE TABLE splink_evidencia AS
        SELECT
            CASE WHEN p.match_probability < {UMBRAL_ALTO} AND p.match_probability >= 0.9
                      THEN 'revisar alto'
                 WHEN p.match_probability < 0.9 THEN 'revisar'
                 WHEN pc.veredicto = 'grupo_empresarial' THEN 'grupo empresarial'
                 ELSE 'fusionar' END                      AS decision,
            round(p.match_probability, 4)                 AS probabilidad,
            round(p.match_weight, 2)                      AS peso_bits,
            p.unique_id_l, p.unique_id_r,
            l.nombre AS nombre_l, rgt.nombre AS nombre_r,
            l.documento AS doc_l, rgt.documento AS doc_r,
            CASE WHEN l.documento = rgt.documento THEN 'identico'
                 WHEN l.documento = substr(rgt.documento, 1, length(rgt.documento) - 1)
                   OR rgt.documento = substr(l.documento, 1, length(l.documento) - 1)
                      THEN 'digito de verificacion'
                 WHEN l.documento IS NULL OR rgt.documento IS NULL THEN 'sin documento'
                 ELSE 'distinto' END                      AS relacion_documento,
            l.tipo_nodo AS tipo_l, rgt.tipo_nodo AS tipo_r,
            l.representante_legal AS rl_l, rgt.representante_legal AS rl_r,
            l.n_contratos_2025 AS contratos_l, rgt.n_contratos_2025 AS contratos_r,
            l.valor_contratos_2025 AS valor_l, rgt.valor_contratos_2025 AS valor_r
        FROM splink_pares p
        JOIN par_clasificado pc ON pc.unique_id_l = p.unique_id_l
                               AND pc.unique_id_r = p.unique_id_r
        JOIN nodo_2025 l   ON l.id_nodo   = p.unique_id_l
        JOIN nodo_2025 rgt ON rgt.id_nodo = p.unique_id_r
    """)
    publicar(con, "splink_evidencia", r["pq"])

    print(con.execute("""
        SELECT decision, count(*) AS pares,
               count(*) FILTER (WHERE relacion_documento = 'digito de verificacion') AS por_dv,
               count(*) FILTER (WHERE relacion_documento = 'identico') AS doc_identico,
               count(*) FILTER (WHERE relacion_documento = 'sin documento') AS sin_doc
        FROM splink_evidencia GROUP BY 1 ORDER BY 1""").df().to_string(index=False))
    print()
    print(con.execute("""
        SELECT count(*) FILTER (WHERE fusionado) AS nodos_fusionados,
               count(DISTINCT id_nodo_resuelto) FILTER (WHERE fusionado) AS grupos,
               count(*) AS nodos_totales,
               count(DISTINCT id_nodo_resuelto) AS actores_resueltos
        FROM splink_cluster""").df().to_string(index=False))
    con.close()


def registrar(r):
    """Publica las tablas nuevas como vistas en la base de red."""
    out = duckdb.connect(r["bd"])
    for t in ("splink_entrada", "splink_parametros", "splink_pares",
              "splink_cluster", "splink_evidencia", "splink_grupo_empresarial"):
        fs = parquets(os.path.join(r["pq"], t))
        if fs:
            out.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM "
                        f"read_parquet({lista_sql(fs)}, union_by_name=true)")
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["preparar", "entrenar", "predecir",
                                      "agrupar", "registrar", "todo"])
    ap.add_argument("--raiz", default=RAIZ)
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--max-pares", type=float, default=5e6, dest="max_pares")
    a = ap.parse_args()
    r = rutas(a.raiz)
    os.makedirs(r["sp"], exist_ok=True)

    etapas = (["preparar", "entrenar", "predecir", "agrupar", "registrar"]
              if a.etapa == "todo" else [a.etapa])
    for e in etapas:
        print(f"\n[{e}]")
        if e == "preparar":
            etapa_preparar(r, a.ram, a.hilos)
        elif e == "entrenar":
            etapa_entrenar(r, a.ram, a.hilos, int(a.max_pares))
        elif e == "predecir":
            etapa_predecir(r, a.ram, a.hilos)
        elif e == "agrupar":
            etapa_agrupar(r, a.ram, a.hilos)
        elif e == "registrar":
            registrar(r)
            print("  vistas registradas en secop_tfm_red2025.duckdb")


if __name__ == "__main__":
    main()
