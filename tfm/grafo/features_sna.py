#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcula las features de red por nodo para el capítulo 5.

Produce los tres bloques de variables que emplea la comparativa de modelos:

  (i)   topológicas: grado, grado ponderado (strength), PageRank, comunidad
        Louvain y pertenencia a la componente gigante;
  (ii)  volumen y diversificación: número de contratos, valor adjudicado,
        número de entidades distintas y valor medio, con transformación log10;
  (iii) indicadores CRI de Fazekas, que ya calcula la vista v_cri_proveedor.

El grafo se construye sobre la proyección bipartita entidad-proveedor, que es
la que emplea el analisis, a partir de las aristas `adjudica` de vinculo_2025.

Salida: salidas/modelos/features_nodo.parquet

Uso:
    python -m tfm.grafo.features_sna --ram 12GB
"""

import argparse
import os
import sys

import duckdb
import networkx as nx
import pandas as pd

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
SALIDA = rutas.resultados()

TABLAS = ["vinculo_2025", "nodo_2025", "nodo_cuenta", "contrato_2025",
          "proceso_competencia", "participacion_2025"]


def abrir(ram):
    """Vistas sobre OUTPUTS/gold. No se usa el .duckdb porque guarda rutas
    absolutas y deja de funcionar al mover el proyecto de equipo."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{ram}'")
    con.execute("SET preserve_insertion_order=false")
    for t in TABLAS:
        patron = os.path.join(PQ, t, "*.parquet").replace("\\", "/")
        if not os.path.isdir(os.path.join(PQ, t)):
            sys.exit(f"falta OUTPUTS/gold/{t}")
        con.execute(f"CREATE OR REPLACE VIEW {t} AS "
                    f"SELECT * FROM read_parquet('{patron}')")
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram", default="6GB")
    a = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)
    con = abrir(a.ram)

    # ------------------------------------------------------------------
    # 1. Grafo bipartito entidad-proveedor
    # ------------------------------------------------------------------
    print("[1] grafo bipartito entidad-proveedor")
    aristas = con.execute("""
        SELECT origen AS entidad, destino AS nodo, peso, coalesce(valor, 0) AS valor
        FROM vinculo_2025 WHERE tipo_vinculo = 'adjudica'
    """).df()
    print(f"    aristas: {len(aristas):,}")

    G = nx.Graph()
    for e, n, p, v in aristas.itertuples(index=False):
        G.add_edge(e, n, weight=float(p))
    print(f"    nodos: {G.number_of_nodes():,} | aristas: {G.number_of_edges():,}")

    # ------------------------------------------------------------------
    # 2. Métricas topológicas
    # ------------------------------------------------------------------
    print("[2] métricas topológicas")
    grado = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    print("    pagerank…")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

    print("    componente gigante…")
    componentes = sorted(nx.connected_components(G), key=len, reverse=True)
    gigante = componentes[0] if componentes else set()

    print("    comunidades Louvain…")
    comunidades = nx.community.louvain_communities(G, seed=42, weight="weight")
    de_comunidad = {n: i for i, c in enumerate(comunidades) for n in c}
    modularidad = nx.community.modularity(G, comunidades, weight="weight")
    print(f"    {len(comunidades):,} comunidades · modularidad {modularidad:.4f}")

    topo = pd.DataFrame({
        "id_nodo": list(G.nodes()),
        "grado": [grado.get(n, 0) for n in G.nodes()],
        "strength": [strength.get(n, 0.0) for n in G.nodes()],
        "pagerank": [pagerank.get(n, 0.0) for n in G.nodes()],
        "comunidad": [de_comunidad.get(n, -1) for n in G.nodes()],
        "en_componente_gigante": [n in gigante for n in G.nodes()],
    })
    topo = topo[~topo["id_nodo"].str.startswith("ENT:")]
    print(f"    nodos proveedor con métricas: {len(topo):,}")

    con.register("topo", topo)

    # ------------------------------------------------------------------
    # 3. Volumen, diversificación y CRI
    # ------------------------------------------------------------------
    print("[3] volumen, diversificación y CRI")
    con.execute("""
        CREATE OR REPLACE TABLE features_nodo AS
        WITH contratos AS (
            SELECT k.id_nodo,
                   count(DISTINCT c.id_contrato)     AS n_contratos,
                   count(DISTINCT c.codigo_entidad)  AS n_entidades,
                   sum(c.valor_del_contrato)         AS valor_total,
                   avg(c.valor_del_contrato)         AS valor_medio
            FROM contrato_2025 c
            JOIN nodo_cuenta k ON k.codigo_proveedor = c.codigo_proveedor
            WHERE NOT c.es_outlier_valor
            GROUP BY 1
        ),
        competencia AS (
            -- señales del CRI: fracción de procesos con oferente único y por
            -- modalidad directa o de mínima cuantía
            SELECT k.id_nodo,
                   count(*)                                                   AS n_procesos,
                   avg(CASE WHEN pc.n_oferentes <= 1 THEN 1.0 ELSE 0 END)     AS pct_single
            FROM participacion_2025 p
            JOIN nodo_cuenta k ON k.codigo_proveedor = p.codigo_proveedor
            JOIN proceso_competencia pc ON pc.id_portafolio = p.id_portafolio
            WHERE p.fue_adjudicado
            GROUP BY 1
        ),
        modalidad AS (
            SELECT k.id_nodo,
                   avg(CASE WHEN lower(c.modalidad_de_contratacion) LIKE '%directa%'
                            THEN 1.0 ELSE 0 END)                              AS pct_directa,
                   avg(CASE WHEN lower(c.modalidad_de_contratacion) LIKE '%m_nima cuant%'
                            THEN 1.0 ELSE 0 END)                              AS pct_minima
            FROM contrato_2025 c
            JOIN nodo_cuenta k ON k.codigo_proveedor = c.codigo_proveedor
            GROUP BY 1
        )
        SELECT
            n.id_nodo,
            n.tipo_nodo,
            n.n_cuentas_plataforma,
            coalesce(t.grado, 0)                       AS grado,
            coalesce(t.strength, 0)                    AS strength,
            coalesce(t.pagerank, 0)                    AS pagerank,
            coalesce(t.comunidad, -1)                  AS comunidad,
            coalesce(t.en_componente_gigante, FALSE)   AS en_componente_gigante,
            coalesce(ct.n_contratos, 0)                AS n_contratos,
            coalesce(ct.n_entidades, 0)                AS n_entidades,
            coalesce(ct.valor_total, 0)                AS valor_total,
            coalesce(ct.valor_medio, 0)                AS valor_medio,
            log10(coalesce(ct.n_contratos, 0) + 1)     AS log_n_contratos,
            log10(coalesce(ct.valor_total, 0) + 1)     AS log_valor_total,
            log10(coalesce(ct.valor_medio, 0) + 1)     AS log_valor_medio,
            coalesce(cp.n_procesos, 0)                 AS n_procesos,
            coalesce(cp.pct_single, 0)                 AS pct_single,
            coalesce(md.pct_directa, 0)                AS pct_directa,
            coalesce(md.pct_minima, 0)                 AS pct_minima,
            -- concentración: 1 - entidades/contratos
            CASE WHEN coalesce(ct.n_contratos, 0) > 0
                 THEN 1 - (ct.n_entidades::DOUBLE / ct.n_contratos)
                 ELSE 0 END                            AS concentracion_entidades,
            least(coalesce(ct.n_contratos, 0) / 50.0, 1.0) AS recurrencia
        FROM nodo_2025 n
        LEFT JOIN topo        t  ON t.id_nodo  = n.id_nodo
        LEFT JOIN contratos   ct ON ct.id_nodo = n.id_nodo
        LEFT JOIN competencia cp ON cp.id_nodo = n.id_nodo
        LEFT JOIN modalidad   md ON md.id_nodo = n.id_nodo
        WHERE coalesce(ct.n_contratos, 0) > 0          -- adjudicatarios
    """)

    # Índice de Riesgo de Corrupción de Fazekas, con los pesos publicados
    con.execute("""
        CREATE OR REPLACE TABLE features_nodo AS
        SELECT *,
               0.30 * pct_single
             + 0.20 * pct_directa
             + 0.20 * concentracion_entidades
             + 0.15 * recurrencia
             + 0.15 * pct_minima                       AS cri_score
        FROM features_nodo
    """)
    con.execute("""
        CREATE OR REPLACE TABLE features_nodo AS
        SELECT *, cri_score >= (SELECT quantile_cont(cri_score, 0.9)
                                FROM features_nodo)    AS riesgo_alto
        FROM features_nodo
    """)

    n, alto, umbral = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE riesgo_alto),
               min(cri_score) FILTER (WHERE riesgo_alto) FROM features_nodo
    """).fetchone()
    print(f"    proveedores con features : {n:,}")
    print(f"    riesgo alto (p90)        : {alto:,}")
    print(f"    umbral CRI               : {umbral:.4f}")

    # ------------------------------------------------------------------
    # 4. Publicación y topología del grafo para la Tabla 5.3
    # ------------------------------------------------------------------
    destino = os.path.join(SALIDA, "features_nodo.parquet").replace("\\", "/")
    #: El `ORDER BY` no es cosmético. Un motor de base de datos no se
    #: compromete a devolver las filas de una unión en ningún orden concreto:
    #: reparte el trabajo entre hilos y emite lo que va terminando. Este módulo
    #: además desactiva `preserve_insertion_order` para acotar la memoria, con
    #: lo que ese orden queda expresamente indefinido.
    #:
    #: Aguas abajo, la partición se obtiene con `train_test_split` sobre
    #: índices posicionales: las mismas filas escritas en otro orden producen
    #: otro conjunto de prueba y, con él, otras métricas. Ordenar por la clave
    #: al escribir cuesta una ordenación de ochenta y tres mil filas y hace que
    #: dos regeneraciones del fichero den exactamente el mismo resultado.
    con.execute(f"COPY (SELECT * FROM features_nodo ORDER BY id_nodo) "
                f"TO '{destino}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    topologia = pd.DataFrame([
        ("Nodos totales", G.number_of_nodes()),
        ("Nodos entidad", sum(1 for x in G.nodes() if str(x).startswith("ENT:"))),
        ("Nodos proveedor", sum(1 for x in G.nodes() if not str(x).startswith("ENT:"))),
        ("Aristas únicas", G.number_of_edges()),
        ("Densidad", round(nx.density(G), 8)),
        ("Componentes conexas", len(componentes)),
        ("Componente gigante", len(gigante)),
        ("Comunidades Louvain", len(comunidades)),
        ("Modularidad", round(modularidad, 4)),
        ("Grado medio", round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2)),
        ("Grado máximo", max(grado.values()) if grado else 0),
    ], columns=["metrica", "valor"])
    topologia.to_csv(os.path.join(SALIDA, "topologia_grafo.csv"), index=False)

    print(f"\n[4] topología del grafo")
    print(topologia.to_string(index=False))
    print(f"\nOK -> {destino}")
    con.close()


if __name__ == "__main__":
    main()
