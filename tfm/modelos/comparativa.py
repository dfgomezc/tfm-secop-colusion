#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparativa entre análisis de redes, aprendizaje automático tabular y GNN.

Entrena y evalúa cuatro modelos sobre la misma partición y la misma etiqueta:

  - línea SNA : Regresión Logística sobre métricas de red
  - líneas ML : Random Forest y Gradient Boosting
  - línea relacional : SGC (Wu et al., 2019). La propagación de características
                a 1 y 2 saltos se precomputa por productos matriciales dispersos
                sobre la incidencia entidad-proveedor, y alimenta un MLP. NO es
                un GraphSAGE: no hay agregadores aprendidos ni paso de mensajes.

Pseudo-etiqueta: riesgo alto según el CRI de Fazekas (percentil 90). Para
evitar fuga entre etiqueta y variables, los modelos NO reciben las componentes
del CRI: solo topología y volumen.

Partición estratificada 70/30 con semilla 42, búsqueda de hiperparámetros por
validación cruzada estratificada y métrica de selección AUC-ROC.

Cada modelo guarda sus predicciones, de modo que el script se puede ejecutar
por partes y reanudar sin reentrenar lo ya hecho.

Salidas en OUTPUTS/:
    tabla_5_9_logistica.csv … tabla_5_12_gnn.csv   rejillas exploradas
    tabla_5_13_comparativa.csv                     resultados en test
    importancia_features_rf.csv                    Figura 5.11
    curvas_roc.csv, curvas_pr.csv                  Figuras 5.9 y 5.10

Uso:
    python -m tfm.modelos.comparativa                 # todos
    python -m tfm.modelos.comparativa --solo gb       # uno concreto
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tfm import rutas

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
MODELOS = rutas.resultados()

#: Variables que ven los modelos. Se excluyen a propósito pct_single,
#: pct_directa, pct_minima, concentracion_entidades y recurrencia: son las
#: componentes del CRI, que es la etiqueta, e incluirlas sería fuga.
FEATURES = [
    "grado", "strength", "pagerank", "en_componente_gigante",
    "n_contratos", "n_entidades", "log_n_contratos", "log_valor_total",
    "log_valor_medio", "n_procesos", "n_cuentas_plataforma",
]

ETIQUETAS = {
    "lr": "SNA - Regresión Logística",
    "rf": "ML - Random Forest",
    "gb": "ML - Gradient Boosting",
    "gnn": "SGC - propagación + MLP",
}


# ---------------------------------------------------------------------------
def recall_en_k(y, p, k=200):
    """De los k con mayor puntuación, qué fracción del total de positivos se
    recupera. Es la métrica operativa: mide el rendimiento de una lista corta
    de casos priorizados para auditoría."""
    orden = np.argsort(-p)[:k]
    return float(y[orden].sum() / max(y.sum(), 1))


def metricas(y, p, umbral=0.5):
    return {
        "AUC-ROC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, (p >= umbral).astype(int), zero_division=0),
        "Recall@200": recall_en_k(y, p, 200),
    }


def ruta_pred(clave):
    return os.path.join(MODELOS, f"_pred_{clave}.csv")


def rejilla(nombre, estimador, malla, X, y, cv):
    n = int(np.prod([len(v) for v in malla.values()]))
    print(f"\n[{nombre}] explorando {n} combinaciones…", flush=True)
    gs = GridSearchCV(estimador, malla, scoring="roc_auc", cv=cv, n_jobs=2)
    gs.fit(X, y)
    r = pd.DataFrame(gs.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")]
    tabla = r[cols + ["mean_test_score", "std_test_score"]].copy()
    tabla.columns = [c.replace("param_", "").replace("clf__", "")
                     for c in tabla.columns]
    tabla = tabla.rename(columns={"mean_test_score": "CV AUC",
                                  "std_test_score": "sigma"})
    tabla["CV AUC"] = tabla["CV AUC"].round(4)
    tabla["sigma"] = tabla["sigma"].round(3)
    tabla = tabla.sort_values("CV AUC", ascending=False).reset_index(drop=True)
    print(f"    mejor: {gs.best_params_}  CV AUC={gs.best_score_:.4f}", flush=True)
    return gs.best_estimator_, tabla


def vecindario(ids, features_norm, aristas):
    """Agregación GraphSAGE-like sin librería de grafos: se construye la
    incidencia proveedor-entidad B y se usa A = B·Bᵀ, que conecta a los
    proveedores que comparten comprador. h1 promedia el vecindario a un salto
    y h2 lo propaga a dos."""
    idx = {n: i for i, n in enumerate(ids)}
    ent, filas, cols = {}, [], []
    for e, n in aristas:
        if n not in idx:
            continue
        filas.append(idx[n])
        cols.append(ent.setdefault(e, len(ent)))
    B = sparse.csr_matrix((np.ones(len(filas)), (filas, cols)),
                          shape=(len(ids), len(ent)))
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    g = np.asarray(A.sum(axis=1)).ravel()
    g[g == 0] = 1
    h1 = np.asarray(A @ features_norm) / g[:, None]
    h2 = np.asarray(A @ h1) / g[:, None]
    return np.hstack([features_norm, h1, h2])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--solo", choices=list(ETIQUETAS), default=None,
                    help="entrena solo ese modelo y guarda su resultado")
    ap.add_argument("--rehacer", action="store_true")
    a = ap.parse_args()
    os.makedirs(MODELOS, exist_ok=True)

    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.ram}'")
    ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{ruta}')").df()
    pq = rutas.parquet_de("vinculo_2025", "*.parquet").replace("\\", "/")
    aristas = con.execute(f"SELECT origen, destino FROM read_parquet('{pq}') "
                          "WHERE tipo_vinculo = 'adjudica'").fetchall()
    con.close()

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    X = df[FEATURES].astype(float).values
    y = df["riesgo_alto"].astype(int).values
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA, stratify=y)
    print(f"proveedores {len(df):,} · positivos {y.sum():,} "
          f"({100*y.mean():.1f} %) · test {len(i_te):,} "
          f"({y[i_te].sum():,} positivos)")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
    pendiente = [k for k in ETIQUETAS if a.solo in (None, k)]

    def toca(clave):
        if clave not in pendiente:
            return False
        if os.path.exists(ruta_pred(clave)) and not a.rehacer:
            print(f"\n[{ETIQUETAS[clave]}] ya calculado")
            return False
        return True

    # ---- 1. SNA: Regresión Logística -------------------------------------
    if toca("lr"):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    random_state=SEMILLA))])
        mejor, tabla = rejilla(ETIQUETAS["lr"], pipe,
                               {"clf__C": [0.01, 0.1, 1.0, 10.0],
                                "clf__class_weight": [None, "balanced"]},
                               X[i_tr], y[i_tr], cv)
        tabla.to_csv(os.path.join(MODELOS, "tabla_5_9_logistica.csv"), index=False)
        p = mejor.predict_proba(X[i_te])[:, 1]
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(ruta_pred("lr"), index=False)

    # ---- 2. ML: Random Forest --------------------------------------------
    if toca("rf"):
        mejor, tabla = rejilla(ETIQUETAS["rf"],
                               RandomForestClassifier(random_state=SEMILLA, n_jobs=2),
                               {"n_estimators": [100, 200], "max_depth": [8, 12],
                                "class_weight": [None, "balanced"]},
                               X[i_tr], y[i_tr], cv)
        tabla.to_csv(os.path.join(MODELOS, "tabla_5_10_random_forest.csv"),
                     index=False)
        p = mejor.predict_proba(X[i_te])[:, 1]
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(ruta_pred("rf"), index=False)
        imp = pd.DataFrame({"feature": FEATURES,
                            "importancia": mejor.feature_importances_})
        imp.sort_values("importancia", ascending=False).to_csv(
            os.path.join(MODELOS, "importancia_features_rf.csv"), index=False)

    # ---- 3. ML: Gradient Boosting ----------------------------------------
    if toca("gb"):
        mejor, tabla = rejilla(ETIQUETAS["gb"],
                               GradientBoostingClassifier(random_state=SEMILLA),
                               {"n_estimators": [100],
                                "learning_rate": [0.05, 0.10],
                                "max_depth": [3, 4]},
                               X[i_tr], y[i_tr], cv)
        tabla.to_csv(os.path.join(MODELOS, "tabla_5_11_gradient_boosting.csv"),
                     index=False)
        p = mejor.predict_proba(X[i_te])[:, 1]
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(ruta_pred("gb"), index=False)

    # ---- 4. SGC: propagación de características + MLP ---------------------
    if toca("gnn"):
        print("\n[SGC] precomputando la propagación a 1 y 2 saltos…",
              flush=True)
        Xn = StandardScaler().fit_transform(X)
        Xg = vecindario(df["id_nodo"].tolist(), Xn, aristas)
        print(f"    embedding de {Xg.shape[1]} dimensiones", flush=True)
        mejor, tabla = rejilla(ETIQUETAS["gnn"],
                               MLPClassifier(max_iter=250, early_stopping=True,
                                             random_state=SEMILLA),
                               {"hidden_layer_sizes": [(128, 64), (64,)],
                                "alpha": [1e-4, 1e-3]},
                               Xg[i_tr], y[i_tr], cv)
        tabla.to_csv(os.path.join(MODELOS, "tabla_5_12_gnn.csv"), index=False)
        p = mejor.predict_proba(Xg[i_te])[:, 1]
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(ruta_pred("gnn"), index=False)

    # ---- comparativa final -----------------------------------------------
    listos = [k for k in ETIQUETAS if os.path.exists(ruta_pred(k))]
    if len(listos) < len(ETIQUETAS):
        faltan = [ETIQUETAS[k] for k in ETIQUETAS if k not in listos]
        print(f"\n[!] faltan por entrenar: {', '.join(faltan)}")
        return

    filas, roc, pr = [], [], []
    for k in ETIQUETAS:
        d = pd.read_csv(ruta_pred(k))
        yy, pp = d["y"].values, d["p"].values
        filas.append({"Modelo": ETIQUETAS[k], **metricas(yy, pp)})
        f, t, _ = roc_curve(yy, pp)
        roc.append(pd.DataFrame({"modelo": ETIQUETAS[k], "fpr": f, "tpr": t}))
        prec, rec, _ = precision_recall_curve(yy, pp)
        pr.append(pd.DataFrame({"modelo": ETIQUETAS[k],
                                "precision": prec, "recall": rec}))

    comp = pd.DataFrame(filas).round(4)
    comp.to_csv(os.path.join(MODELOS, "tabla_5_13_comparativa.csv"), index=False)
    pd.concat(roc).to_csv(os.path.join(MODELOS, "curvas_roc.csv"), index=False)
    pd.concat(pr).to_csv(os.path.join(MODELOS, "curvas_pr.csv"), index=False)

    print("\n" + "=" * 66)
    print("TABLA 5.13 — Comparativa final sobre el conjunto de prueba")
    print("=" * 66)
    print(comp.to_string(index=False))
    print(f"\nOK -> {MODELOS}")


if __name__ == "__main__":
    main()
