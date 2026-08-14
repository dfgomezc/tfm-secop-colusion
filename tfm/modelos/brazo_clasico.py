#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluación de las técnicas clásicas de red como detectores.

En la comparativa de modelos, Louvain, PageRank y el índice de riesgo actúan
como ingeniería de variables y no como detectores medidos por derecho propio.
Este módulo cubre esa diferencia: construye tres ordenamientos que no entrenan
nada y los evalúa con las mismas métricas y sobre el mismo conjunto de prueba
que los modelos entrenados.

  1. PageRank puro.
  2. Recuento normalizado de banderas rojas estructurales.
  3. Número de contratos, como línea base trivial.

Salidas, por tema de OUTPUTS/:
    tablas/tabla_5_20_brazo_clasico.csv        los tres ordenamientos clásicos
    tablas/tabla_5_21_siete_ordenamientos.csv  los siete, clásicos y entrenados
    graph_sna/banderas_rojas.csv               la señal por proveedor

Uso:
    python -m tfm.modelos.brazo_clasico
"""

import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from tfm import rutas
from tfm.grafo import triangulos

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
MODELOS = rutas.resultados()

#: Mismo conjunto de variables que emplea la comparativa, solo para reproducir
#: exactamente el mismo `dropna` y, con él, la misma partición.
FEATURES_V0 = [
    "grado", "strength", "pagerank", "en_componente_gigante",
    "n_contratos", "n_entidades", "log_n_contratos", "log_valor_total",
    "log_valor_medio", "n_procesos", "n_cuentas_plataforma",
]

#: La exclusión de banca y seguros y la definición del triángulo viven en
#: `tfm.grafo.triangulos`, que es de donde las toman también el solapamiento y
#: las figuras de red.


# ---------------------------------------------------------------------------
def banderas_rojas(con, ids):
    """Cuatro indicadores binarios por proveedor, normalizados a [0, 1].

    Ninguno procede de un modelo: los cuatro son señales que un analista puede
    enunciar en una frase y verificar contra el expediente.

      t  el actor aparece en al menos un triángulo de riesgo entidad-UT-socio
      c  comparte algún dato de contacto con otro actor
      r  comparte representante legal con otro actor
      a  pertenece a un componente conexo cerrado, es decir, no al gigante

    La señal es la media simple de los cuatro: (t + c + r + a) / 4.
    """
    #: Las vistas del triángulo las crea `tfm.grafo.triangulos`, que es donde
    #: vive la definición. Deja además `exc` y `v` disponibles, que es sobre lo
    #: que se levantan las otras tres banderas.
    triangulos.preparar(con, PQ)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW en_triangulo AS
          SELECT ut AS id_nodo FROM tri
          UNION SELECT socio FROM tri;

        CREATE OR REPLACE TEMP VIEW contacto AS
          SELECT origen AS id_nodo FROM v WHERE tipo_vinculo = 'comparte_contacto'
          UNION SELECT destino FROM v WHERE tipo_vinculo = 'comparte_contacto';

        -- Representante compartido: el nodo de representante legal tiene que
        -- estar enlazado a dos o más actores distintos.
        CREATE OR REPLACE TEMP VIEW rl_compartido AS
          SELECT origen AS id_nodo FROM v
          WHERE tipo_vinculo = 'representado_por'
            AND destino IN (SELECT destino FROM v
                            WHERE tipo_vinculo = 'representado_por'
                            GROUP BY destino HAVING count(DISTINCT origen) > 1);
    """)
    t = {r[0] for r in con.execute("SELECT id_nodo FROM en_triangulo").fetchall()}
    c = {r[0] for r in con.execute("SELECT id_nodo FROM contacto").fetchall()}
    r = {r[0] for r in con.execute("SELECT id_nodo FROM rl_compartido").fetchall()}
    return (np.array([n in t for n in ids], dtype=float),
            np.array([n in c for n in ids], dtype=float),
            np.array([n in r for n in ids], dtype=float))


# ---------------------------------------------------------------------------
def precision_en_k(y, p, k, rng):
    """Precisión sobre los k primeros, con empates deshechos al azar.

    Las banderas rojas solo toman cinco valores distintos, así que en el corte
    de los 200 primeros hay siempre un empate masivo. Deshacerlo por el orden
    de fila daría un resultado arbitrario y no reproducible entre ejecuciones;
    deshacerlo al azar con semilla fija es arbitrario pero sí reproducible, y
    además insesgado respecto del grupo empatado.
    """
    orden = np.lexsort((rng.random(len(p)), -p))[:k]
    return float(y[orden].mean())


def recall_en_k(y, p, k, rng):
    orden = np.lexsort((rng.random(len(p)), -p))[:k]
    return float(y[orden].sum() / max(y.sum(), 1))


def metricas(y, p, umbral=None):
    rng = np.random.default_rng(SEMILLA)
    #: Un ordenamiento no supervisado no produce probabilidades, así que el F1
    #: se calcula sobre el decil superior, que es la prevalencia de la clase
    #: positiva. Para los modelos entrenados el umbral es 0,5, y así se declara.
    if umbral is None:
        corte = np.quantile(p, 0.90)
        pred = (p >= corte).astype(int)
    else:
        pred = (p >= umbral).astype(int)
    return {
        "AUC-ROC": round(roc_auc_score(y, p), 4),
        "PR-AUC": round(average_precision_score(y, p), 4),
        "F1": round(f1_score(y, pred, zero_division=0), 4),
        "Precision@200": round(precision_en_k(y, p, 200, rng), 4),
        "Recall@200": round(recall_en_k(y, p, 200, rng), 4),
    }


# ---------------------------------------------------------------------------
def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{ruta}')").df()
    df = df.dropna(subset=FEATURES_V0).reset_index(drop=True)

    #: Misma partición que la comparativa: mismo orden de filas, mismo
    #: test_size, misma semilla y misma estratificación.
    y = df["riesgo_alto"].astype(int).values
    _, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                               random_state=SEMILLA, stratify=y)
    print(f"conjunto de prueba: {len(i_te):,} nodos, {y[i_te].sum():,} positivos")

    ids = df["id_nodo"].tolist()
    t, c, r = banderas_rojas(con, ids)
    a = 1.0 - df["en_componente_gigante"].astype(float).values
    señal = (t + c + r + a) / 4.0

    pd.DataFrame({"id_nodo": ids, "en_triangulo": t, "comparte_contacto": c,
                  "representante_compartido": r, "componente_cerrado": a,
                  "banderas_rojas": señal}).to_csv(
        os.path.join(MODELOS, "banderas_rojas.csv"), index=False)

    print("\nreparto de la señal de banderas rojas (universo completo)")
    for v, n in sorted(pd.Series(señal).value_counts().items()):
        print(f"  {v:.2f} -> {n:,} proveedores")

    #: Diagnóstico por bandera: si la señal agregada no discrimina, hay que
    #: saber si alguna de las cuatro sí lo hace y las demás la anulan.
    print("\nAUC de cada bandera por separado, sobre el conjunto de prueba")
    diag = []
    for nombre, v in (("en_triangulo", t), ("comparte_contacto", c),
                      ("representante_compartido", r), ("componente_cerrado", a)):
        auc = roc_auc_score(y[i_te], v[i_te])
        tasa = float(v.mean())
        print(f"  {nombre:26s} AUC {auc:.4f}   marca al {100*tasa:5.1f} % del universo")
        diag.append({"bandera": nombre, "AUC-ROC": round(auc, 4),
                     "% del universo marcado": round(100 * tasa, 2)})
    pd.DataFrame(diag).to_csv(
        os.path.join(MODELOS, "tabla_5_22_banderas_diagnostico.csv"), index=False)

    ordenamientos = {
        "Clásico - PageRank puro": df["pagerank"].astype(float).values,
        "Clásico - Banderas rojas estructurales": señal,
        "Trivial - Número de contratos": df["n_contratos"].astype(float).values,
    }
    filas = []
    for nombre, p in ordenamientos.items():
        filas.append({"Ordenamiento": nombre, "Tipo": "clásico",
                      **metricas(y[i_te], p[i_te])})
    t_clasico = pd.DataFrame(filas)
    t_clasico.to_csv(os.path.join(MODELOS, "tabla_5_20_brazo_clasico.csv"),
                     index=False)
    print("\n" + t_clasico.to_string(index=False))

    #: Los cuatro modelos entrenados, del control de fuga (siete variables).
    ENTRENADOS = {"lr": "SNA - Regresión Logística",
                  "rf": "ML - Random Forest",
                  "gb": "ML - Gradient Boosting",
                  "gnn": "SGC - propagación + MLP"}
    for k, nombre in ENTRENADOS.items():
        f = rutas.artefacto(os.path.join("v1", f"_pred_{k}.csv"))
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        filas.append({"Ordenamiento": nombre, "Tipo": "entrenado",
                      **metricas(d["y"].values, d["p"].values, umbral=0.5)})

    siete = pd.DataFrame(filas)
    siete.to_csv(os.path.join(MODELOS, "tabla_5_21_siete_ordenamientos.csv"),
                 index=False)
    print("\n" + siete.to_string(index=False))


if __name__ == "__main__":
    main()
