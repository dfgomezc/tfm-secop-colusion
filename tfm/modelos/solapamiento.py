#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""¿Señalan los dos paradigmas a los mismos actores?.

El trabajo recomienda una arquitectura híbrida —modelo relacional para ordenar,
señales estructurales para explicar— pero lo hace con un argumento de
conveniencia. Este script mide el solapamiento entre las dos salidas.

Si es bajo, los dos paradigmas señalan actores distintos y la recomendación deja
de ser una preferencia para pasar a ser una conclusión con evidencia. Si es
alto, la señal estructural es redundante y la arquitectura híbrida solo se
justifica por interpretabilidad. Cualquiera de los dos es un resultado.

Además corrige un defecto de trazabilidad: la partición se persiste con su
`id_nodo`, de modo que cada fila de `_pred_*.csv` puede volver a asociarse a su
actor.

Salidas en OUTPUTS/:
    particion_prueba.csv                 id_nodo y etiqueta del conjunto de prueba
    tabla_5_23_solapamiento.csv          Jaccard, intersección y coberturas
    tabla_5_24_zonas.csv                 perfil de las tres zonas
    solapamiento_actores.csv             a qué zona pertenece cada actor

Uso:
    python -m tfm.modelos.solapamiento
"""

import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tfm import rutas
from tfm.grafo import triangulos

SEMILLA = 42
K = 200
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
MODELOS = rutas.resultados()

FEATURES_V0 = [
    "grado", "strength", "pagerank", "en_componente_gigante",
    "n_contratos", "n_entidades", "log_n_contratos", "log_valor_total",
    "log_valor_medio", "n_procesos", "n_cuentas_plataforma",
]

def actores_en_triangulo(con):
    """Uniones temporales y socios implicados en algún triángulo de riesgo.

    La definición y la exclusión de banca y seguros viven en
    `tfm.grafo.triangulos`, de donde las toman también el brazo clásico y las
    figuras de red: si estuvieran escritas aquí volverían a divergir.
    """
    return triangulos.actores(con, PQ)


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def cruzada(en_modelo, en_triangulo, universo):
    """Tabla 2x2 y contraste de independencia sobre el conjunto de prueba."""
    from scipy.stats import chi2_contingency, fisher_exact
    a = len(en_modelo & en_triangulo)
    b = len(en_modelo - en_triangulo)
    c = len(en_triangulo - en_modelo)
    d = len(universo) - a - b - c
    tabla = np.array([[a, b], [c, d]])
    chi2, p, _, esperado = chi2_contingency(tabla)
    odds, p_fisher = fisher_exact(tabla)
    return {"ambos": a, "solo modelo": b, "solo triángulo": c, "ninguno": d,
            "esperados si independientes": round(float(esperado[0][0]), 2),
            "chi2": round(float(chi2), 2), "p (chi2)": f"{p:.3g}",
            "odds ratio": round(float(odds), 3), "p (Fisher)": f"{p_fisher:.3g}"}


def perfil(df, ids, etiqueta):
    """Descriptores medianos de un conjunto de actores."""
    d = df[df["id_nodo"].isin(ids)]
    if d.empty:
        return {"Zona": etiqueta, "Actores": 0}
    return {
        "Zona": etiqueta,
        "Actores": len(d),
        "% uniones temporales": round(100 * (d["tipo_nodo"] == "union_temporal").mean(), 1),
        "Contratos (mediana)": round(float(d["n_contratos"].median()), 1),
        "Entidades (mediana)": round(float(d["n_entidades"].median()), 1),
        "Valor total (mediana, mill.)": round(float(d["valor_total"].median()) / 1e6, 1),
        "PageRank (mediana, 1e-5)": round(float(d["pagerank"].median()) * 1e5, 2),
        "% en componente gigante": round(100 * d["en_componente_gigante"].mean(), 1),
        "% contratación directa (mediana)": round(float(d["pct_directa"].median()), 3),
        "CRI (mediana)": round(float(d["cri_score"].median()), 3),
    }


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{ruta}')").df()
    df = df.dropna(subset=FEATURES_V0).reset_index(drop=True)
    y = df["riesgo_alto"].astype(int).values

    _, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                               random_state=SEMILLA, stratify=y)

    #: Defecto de trazabilidad que este script corrige: hasta ahora las
    #: predicciones se guardaban sin identificador, de modo que no había forma
    #: de saber a qué actor correspondía cada fila.
    prueba = df.iloc[i_te][["id_nodo", "tipo_nodo"]].copy()
    prueba["riesgo_alto"] = y[i_te]
    prueba.to_csv(rutas.resultados("particion_prueba.csv"), index=False)
    ids_te = prueba["id_nodo"].tolist()
    universo = set(ids_te)
    print(f"conjunto de prueba: {len(universo):,} actores, "
          f"{int(prueba['riesgo_alto'].sum()):,} positivos")

    tri = actores_en_triangulo(con)
    tri_te = tri & universo
    print(f"actores en triángulo de riesgo: {len(tri):,} en el universo, "
          f"{len(tri_te):,} en el conjunto de prueba")

    #: Se comparan las dos configuraciones para que la conclusión no dependa de
    #: una partición con fuga: V0 son las once variables, V1 las siete limpias.
    #: Se piden por `rutas.artefacto`, que prefiere lo recalculado en esta
    #: ejecución y, si no lo hay, toma lo publicado. Leyendo solo `salidas/`
    #: este apartado no se podía reproducir desde el paquete de datos.
    fuentes = {"V0": ("", "gnn"), "V1": ("v1", "gnn"),
               "V1-RF": (os.path.join("matriz", "propagadas"), "rf")}
    filas, zonas, asignacion = [], [], {}
    for var, (ruta_v, clave) in fuentes.items():
        f = rutas.artefacto(os.path.join(ruta_v, f"_pred_{clave}.csv"))
        if not os.path.exists(f):
            print(f"  [!] falta {f}")
            continue
        p = pd.read_csv(f)["p"].values
        top = {ids_te[i] for i in np.argsort(-p)[:K]}

        inter = top & tri_te
        filas.append({
            "Configuración": var,
            "Top-K del modelo": len(top),
            "Actores en triángulo (prueba)": len(tri_te),
            "Intersección": len(inter),
            "Jaccard": round(jaccard(top, tri_te), 4),
            "% del top-K que está en triángulo": round(100 * len(inter) / len(top), 1),
            "% de los de triángulo que están en el top-K":
                round(100 * len(inter) / max(len(tri_te), 1), 2),
            **cruzada(top, tri_te, universo),
        })
        if var == "V1":
            zonas = [perfil(df, top - tri_te, "Solo el modelo"),
                     perfil(df, inter, "Ambos"),
                     perfil(df, tri_te - top, "Solo la señal estructural"),
                     perfil(df, universo - top - tri_te, "Ninguno")]
            for n in universo:
                asignacion[n] = ("ambos" if n in top and n in tri_te else
                                 "solo_modelo" if n in top else
                                 "solo_triangulo" if n in tri_te else "ninguno")

    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(MODELOS, "tabla_5_23_solapamiento.csv"), index=False)
    print("\n" + t.to_string(index=False))

    z = pd.DataFrame(zonas)
    z.to_csv(os.path.join(MODELOS, "tabla_5_24_zonas.csv"), index=False)
    print("\n" + z.to_string(index=False))

    pd.DataFrame({"id_nodo": list(asignacion), "zona": list(asignacion.values())}
                 ).to_csv(os.path.join(MODELOS, "solapamiento_actores.csv"),
                          index=False)


if __name__ == "__main__":
    main()
