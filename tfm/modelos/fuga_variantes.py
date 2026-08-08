#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuga entre la pseudo-etiqueta y las variables.

El apartado 5.1.7 afirma que los modelos no reciben las componentes del CRI.
Es cierto en el código y falso en efecto: `grado` es exactamente `n_entidades`,
y `concentracion_entidades` y `recurrencia` se reconstruyen desde `n_contratos`
y `n_entidades`. Entre las dos pesan el 35 % del índice.

Este script mide el efecto y ejecuta dos variantes de control:

  F2.1a  R² de cada componente del CRI predicha desde las once variables.
  V1     misma etiqueta, sin las variables reconstruibles (siete features).
  V2     misma información de entrada, con una etiqueta que no es derivable:
         solo las componentes de modalidad, renormalizadas.

V0 no se toca. Cada variante escribe en su propio subdirectorio para que las
tres puedan compararse.

Uso:
    python -m tfm.modelos.fuga_variantes --fase fuga
    python -m tfm.modelos.fuga_variantes --fase v1 --solo gb
    python -m tfm.modelos.fuga_variantes --fase v2
    python -m tfm.modelos.fuga_variantes --fase tabla     # consolida las tres
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tfm import rutas

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
MODELOS = rutas.resultados()

#: V0: las once variables que recibe hoy la comparativa.
FEATURES_V0 = [
    "grado", "strength", "pagerank", "en_componente_gigante",
    "n_contratos", "n_entidades", "log_n_contratos", "log_valor_total",
    "log_valor_medio", "n_procesos", "n_cuentas_plataforma",
]

#: V1: se retiran las cuatro variables desde las que el índice es
#: reconstruible. `grado` entra en la lista porque en la proyección bipartita
#: con aristas únicas es idéntica a `n_entidades`, no por su significado.
RECONSTRUIBLES = ["n_contratos", "n_entidades", "log_n_contratos", "grado"]
FEATURES_V1 = [f for f in FEATURES_V0 if f not in RECONSTRUIBLES]

#: Componentes del CRI y su peso, tal como los compone 08_features_sna.py.
CRI = {"pct_single": 0.30, "pct_directa": 0.20, "concentracion_entidades": 0.20,
       "recurrencia": 0.15, "pct_minima": 0.15}

#: V2: etiqueta que no es derivable de las variables. Solo las componentes de
#: modalidad, que describen CÓMO se contrató y no CUÁNTO.
MODALIDAD = {"pct_single": 0.30, "pct_directa": 0.20, "pct_minima": 0.15}

ETIQUETAS = {
    "lr": "SNA - Regresión Logística",
    "rf": "ML - Random Forest",
    "gb": "ML - Gradient Boosting",
    "gnn": "SGC - propagación + MLP",
}


# ---------------------------------------------------------------------------
def cargar():
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{ruta}')").df()
    pq = rutas.parquet_de("vinculo_2025", "*.parquet").replace("\\", "/")
    aristas = con.execute(f"SELECT origen, destino FROM read_parquet('{pq}') "
                          "WHERE tipo_vinculo = 'adjudica'").fetchall()
    con.close()
    return df.dropna(subset=FEATURES_V0).reset_index(drop=True), aristas


def dir_variante(v):
    d = os.path.join(MODELOS, v)
    os.makedirs(d, exist_ok=True)
    return d


def precision_en_k(y, p, k=200):
    return float(y[np.argsort(-p)[:k]].mean())


def recall_en_k(y, p, k=200):
    return float(y[np.argsort(-p)[:k]].sum() / max(y.sum(), 1))


def metricas(y, p, umbral=0.5):
    return {
        "AUC-ROC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, (p >= umbral).astype(int), zero_division=0),
        "Precision@200": precision_en_k(y, p),
        "Recall@200": recall_en_k(y, p),
    }


def vecindario(ids, features_norm, aristas):
    """Propagación de características sobre la incidencia entidad-proveedor.

    A = B·Bᵀ conecta a los proveedores que comparten comprador; h1 promedia el
    vecindario a un salto y h2 lo propaga a dos. Es el operador de una
    Simplified Graph Convolution, no un GraphSAGE con agregadores aprendidos.
    """
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
def fase_fuga(df):
    """F2.1a — ¿Cuánto de cada componente del CRI hay ya en las variables?"""
    X = df[FEATURES_V0].astype(float).values
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA)
    filas = []
    for comp, peso in CRI.items():
        y = df[comp].astype(float).values
        lin = LinearRegression().fit(X[i_tr], y[i_tr])
        r2_lin = r2_score(y[i_te], lin.predict(X[i_te]))
        #: El árbol capta relaciones no lineales como los cocientes de los que
        #: se componen dos de las cinco variables.
        arb = RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                    random_state=SEMILLA, n_jobs=2)
        arb.fit(X[i_tr], y[i_tr])
        r2_arb = r2_score(y[i_te], arb.predict(X[i_te]))
        filas.append({"componente": comp, "peso en el CRI": peso,
                      "R2 lineal": round(r2_lin, 4),
                      "R2 no lineal": round(r2_arb, 4)})
        print(f"  {comp:26s} peso {peso:.2f}  R2 lin {r2_lin:6.4f}  "
              f"R2 arbol {r2_arb:6.4f}", flush=True)

    t = pd.DataFrame(filas)
    reconstruible = float((t["peso en el CRI"] * t["R2 no lineal"].clip(0)).sum())
    print(f"\n  fracción del índice reconstruible: {reconstruible:.3f}", flush=True)
    t.to_csv(os.path.join(MODELOS, "tabla_5_17_fuga_cri.csv"), index=False)

    #: Ordenar por la parte exactamente reconstruible, sin entrenar nada.
    y = df["riesgo_alto"].astype(int).values
    parte = (0.20 * df["concentracion_entidades"].astype(float)
             + 0.15 * df["recurrencia"].astype(float)).values
    auc_fuga = roc_auc_score(y, parte)
    print(f"  AUC de la etiqueta usando SOLO esa parte: "
          f"{auc_fuga:.4f}", flush=True)

    #: Estas tres magnitudes se citan en el texto del 5.2.2 y del 5.1.6, así
    #: que necesitan artefacto propio: una cifra que sólo existe en la salida
    #: por consola de una ejecución no es una cifra verificable.
    umbral = float(df["cri_score"].astype(float).quantile(0.90))
    pd.DataFrame([
        {"magnitud": "Umbral del percentil 90 del cri_score",
         "valor": round(umbral, 4)},
        {"magnitud": "AUC-ROC de la ordenación por la parte reconstruible",
         "valor": round(float(auc_fuga), 4)},
        {"magnitud": "Fracción del índice reconstruible desde las variables",
         "valor": round(reconstruible, 4)},
    ]).to_csv(os.path.join(MODELOS, "tabla_5_16_fuga_magnitudes.csv"),
              index=False)
    return t


# ---------------------------------------------------------------------------
def rejilla_incremental(nombre, estimador, malla, X, y, cv, parcial, margen=18):
    """Búsqueda en rejilla con punto de control por combinación.

    El entorno de ejecución interrumpe los procesos largos, así que la búsqueda
    no puede ser atómica: cada combinación se evalúa, se persiste su AUC de
    validación cruzada y la siguiente llamada retoma donde se quedó. Cuando
    todas están evaluadas, reajusta la mejor sobre el conjunto de entrenamiento
    completo y devuelve el estimador.
    """
    import itertools
    import time
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    claves = list(malla)
    combos = [dict(zip(claves, v)) for v in itertools.product(*malla.values())]
    hechas = {}
    if os.path.exists(parcial):
        for _, f in pd.read_csv(parcial).iterrows():
            hechas[f["combo"]] = (float(f["CV AUC"]), float(f["sigma"]))

    t0 = time.time()
    for c in combos:
        etiqueta = repr(sorted(c.items()))
        if etiqueta in hechas:
            continue
        if time.time() - t0 > margen:
            print(f"[{nombre}] {len(hechas)}/{len(combos)} evaluadas; "
                  f"vuelve a lanzar para continuar", flush=True)
            return None, None
        s = cross_val_score(clone(estimador).set_params(**c), X, y,
                            scoring="roc_auc", cv=cv, n_jobs=2)
        hechas[etiqueta] = (float(s.mean()), float(s.std()))
        print(f"    {c} -> CV AUC {s.mean():.4f}", flush=True)
        #: Se persiste tras CADA combinación, no al salir del bucle: si la
        #: llamada se interrumpe a mitad de la siguiente, lo ya calculado no
        #: se pierde.
        pd.DataFrame([{"combo": k, "CV AUC": v[0], "sigma": v[1]}
                      for k, v in hechas.items()]).to_csv(parcial, index=False)

    pd.DataFrame([{"combo": k, "CV AUC": v[0], "sigma": v[1]}
                  for k, v in hechas.items()]).to_csv(parcial, index=False)
    mejor_et = max(hechas, key=lambda k: hechas[k][0])
    mejor_par = dict(eval(mejor_et))
    print(f"[{nombre}] mejor {mejor_par}  CV AUC={hechas[mejor_et][0]:.4f}",
          flush=True)
    modelo = clone(estimador).set_params(**mejor_par).fit(X, y)
    tabla = pd.DataFrame([{**dict(eval(k)), "CV AUC": v[0], "sigma": v[1]}
                          for k, v in hechas.items()]
                         ).sort_values("CV AUC", ascending=False)
    #: Mismo encabezado que produce `rejilla`: los dos caminos escriben el
    #: mismo fichero y tienen que nombrar las columnas igual. Cuando el
    #: estimador es un `Pipeline`, sus parámetros llegan prefijados con el
    #: nombre del paso —`clf__C` en vez de `C`—, y ese prefijo varía con la
    #: versión de scikit-learn, así que se retira aquí y no depende de ella.
    tabla.columns = [c.replace("clf__", "") for c in tabla.columns]
    return modelo, tabla


def rejilla(nombre, estimador, malla, X, y, cv):
    n = int(np.prod([len(v) for v in malla.values()]))
    print(f"\n[{nombre}] {n} combinaciones…", flush=True)
    gs = GridSearchCV(estimador, malla, scoring="roc_auc", cv=cv, n_jobs=2)
    gs.fit(X, y)
    print(f"    mejor {gs.best_params_}  CV AUC={gs.best_score_:.4f}", flush=True)
    r = pd.DataFrame(gs.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")]
    tabla = r[cols + ["mean_test_score", "std_test_score"]].copy()
    tabla.columns = [c.replace("param_", "").replace("clf__", "")
                     for c in tabla.columns]
    tabla = tabla.rename(columns={"mean_test_score": "CV AUC",
                                  "std_test_score": "sigma"})
    return gs.best_estimator_, tabla.sort_values("CV AUC", ascending=False)


def fase_comparativa(df, aristas, variante, solo, rehacer):
    """Ejecuta los cuatro modelos de una variante y persiste sus predicciones."""
    dest = dir_variante(variante)

    if variante == "v1":
        feats, y = FEATURES_V1, df["riesgo_alto"].astype(int).values
    elif variante == "v2":
        feats = FEATURES_V0
        #: Etiqueta alternativa: solo modalidad, renormalizada a suma 1.
        total = sum(MODALIDAD.values())
        score = sum(p * df[c].astype(float) for c, p in MODALIDAD.items()) / total
        #: El índice de solo modalidad está muy empatado: sus tres componentes
        #: son proporciones que valen 1,00 para miles de proveedores, y un corte
        #: por percentil 90 dejaría al 33 % en la clase positiva. Para que la
        #: prevalencia sea comparable con V0 y V1 se toma el decil superior por
        #: rango, con los empates deshechos de forma reproducible.
        rng = np.random.default_rng(SEMILLA)
        orden = np.lexsort((rng.random(len(score)), -score.values))
        y = np.zeros(len(score), dtype=int)
        y[orden[:int(round(0.10 * len(score)))]] = 1
        #: CSV y no parquet: el entorno de ejecución no siempre trae pyarrow,
        #: y esta tabla es pequeña.
        df.assign(cri2=score)[["id_nodo", "cri2"]].to_csv(
            os.path.join(dest, "etiqueta_v2.csv"), index=False)
    else:
        raise SystemExit("variante desconocida")

    X = df[feats].astype(float).values
    print(f"\n=== {variante.upper()} · {len(feats)} variables · "
          f"{y.sum():,} positivos de {len(y):,} ({100*y.mean():.1f} %)", flush=True)
    print(f"    features: {', '.join(feats)}", flush=True)

    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA, stratify=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)

    def pred(k):
        return os.path.join(dest, f"_pred_{k}.csv")

    def toca(k):
        if solo not in (None, k):
            return False
        if os.path.exists(pred(k)) and not rehacer:
            print(f"[{ETIQUETAS[k]}] ya calculado", flush=True)
            return False
        return True

    def guardar(k, p, tabla=None):
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(pred(k), index=False)
        if tabla is not None:
            tabla.round(4).to_csv(os.path.join(dest, f"rejilla_{k}.csv"),
                                  index=False)

    if toca("lr"):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    random_state=SEMILLA))])
        m, t = rejilla_incremental(ETIQUETAS["lr"], pipe,
                                   {"clf__C": [0.01, 0.1, 1.0, 10.0],
                                    "clf__class_weight": [None, "balanced"]},
                                   X[i_tr], y[i_tr], cv,
                                   os.path.join(dest, "_parcial_lr.csv"))
        if m is not None:
            guardar("lr", m.predict_proba(X[i_te])[:, 1], t)

    if toca("rf"):
        m, t = rejilla_incremental(ETIQUETAS["rf"],
                                   RandomForestClassifier(random_state=SEMILLA,
                                                          n_jobs=2),
                                   {"n_estimators": [100, 200],
                                    "max_depth": [8, 12],
                                    "class_weight": [None, "balanced"]},
                                   X[i_tr], y[i_tr], cv,
                                   os.path.join(dest, "_parcial_rf.csv"))
        if m is not None:
            guardar("rf", m.predict_proba(X[i_te])[:, 1], t)
            pd.DataFrame({"feature": feats,
                          "importancia": m.feature_importances_}
                         ).sort_values("importancia", ascending=False).to_csv(
                os.path.join(dest, "importancia_rf.csv"), index=False)

    if toca("gb"):
        m, t = rejilla_incremental(ETIQUETAS["gb"],
                                   GradientBoostingClassifier(random_state=SEMILLA),
                                   {"n_estimators": [100],
                                    "learning_rate": [0.05, 0.10],
                                    "max_depth": [3, 4]},
                                   X[i_tr], y[i_tr], cv,
                                   os.path.join(dest, "_parcial_gb.csv"))
        if m is not None:
            guardar("gb", m.predict_proba(X[i_te])[:, 1], t)

    if toca("gnn"):
        Xn = StandardScaler().fit_transform(X)
        Xg = vecindario(df["id_nodo"].tolist(), Xn, aristas)
        print(f"    embedding de {Xg.shape[1]} dimensiones", flush=True)
        m, t = rejilla_incremental(ETIQUETAS["gnn"],
                                   MLPClassifier(max_iter=250,
                                                 early_stopping=True,
                                                 random_state=SEMILLA),
                                   {"hidden_layer_sizes": [(128, 64), (64,)],
                                    "alpha": [1e-4, 1e-3]},
                                   Xg[i_tr], y[i_tr], cv,
                                   os.path.join(dest, "_parcial_gnn.csv"))
        if m is not None:
            guardar("gnn", m.predict_proba(Xg[i_te])[:, 1], t)

    #: Comparativa de la variante, si ya están los cuatro.
    listos = [k for k in ETIQUETAS if os.path.exists(pred(k))]
    if len(listos) == len(ETIQUETAS):
        filas = []
        for k in ETIQUETAS:
            d = pd.read_csv(pred(k))
            filas.append({"Modelo": ETIQUETAS[k],
                          **{m: round(v, 4) for m, v in
                             metricas(d["y"].values, d["p"].values).items()}})
        t = pd.DataFrame(filas)
        t.to_csv(os.path.join(dest, "comparativa.csv"), index=False)
        print("\n" + t.to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
def fase_matriz(df, aristas, conjunto, solo, rehacer):
    """F2.2 — Separa el efecto de la arquitectura del de la propagación.

    Los tres modelos tabulares reciben hoy las variables propias y el cuarto
    las recibe propagadas: la comparación mezcla arquitectura con ingeniería de
    variables. Aquí se entrenan los CUATRO sobre los DOS conjuntos.

    Se parte de las siete variables que sobreviven al control de fuga, no de
    las once originales: comparar sobre un conjunto con fuga mediría de nuevo
    la capacidad de recomponer el índice.
    """
    dest = dir_variante(os.path.join("matriz", conjunto))
    feats = FEATURES_V1
    y = df["riesgo_alto"].astype(int).values
    X = df[feats].astype(float).values

    if conjunto == "propagadas":
        Xn = StandardScaler().fit_transform(X)
        X = vecindario(df["id_nodo"].tolist(), Xn, aristas)

    print(f"\n=== matriz · {conjunto} · {X.shape[1]} columnas · "
          f"{y.sum():,} positivos de {len(y):,}", flush=True)

    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA, stratify=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)

    def pred(k):
        return os.path.join(dest, f"_pred_{k}.csv")

    def toca(k):
        if solo not in (None, k):
            return False
        if os.path.exists(pred(k)) and not rehacer:
            print(f"[{ETIQUETAS[k]}] ya calculado", flush=True)
            return False
        return True

    def guardar(k, p):
        pd.DataFrame({"y": y[i_te], "p": p}).to_csv(pred(k), index=False)

    mallas = {
        "lr": (Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    random_state=SEMILLA))]),
               {"clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"]}),
        "rf": (RandomForestClassifier(random_state=SEMILLA, n_jobs=2),
               {"n_estimators": [100, 200], "max_depth": [8, 12],
                "class_weight": [None, "balanced"]}),
        "gb": (GradientBoostingClassifier(random_state=SEMILLA),
               {"n_estimators": [100], "learning_rate": [0.05, 0.10],
                "max_depth": [3, 4]}),
        "gnn": (MLPClassifier(max_iter=250, early_stopping=True,
                              random_state=SEMILLA),
                {"hidden_layer_sizes": [(128, 64), (64,)],
                 "alpha": [1e-4, 1e-3]}),
    }
    for k, (est, malla) in mallas.items():
        if not toca(k):
            continue
        m, _ = rejilla_incremental(ETIQUETAS[k], est, malla, X[i_tr], y[i_tr],
                                   cv, os.path.join(dest, f"_parcial_{k}.csv"))
        if m is not None:
            guardar(k, m.predict_proba(X[i_te])[:, 1])


def fase_matriz_tabla():
    """Consolida la matriz {modelo} x {propias, propagadas}."""
    filas = []
    for conjunto in ("propias", "propagadas"):
        ruta = os.path.join(MODELOS, "matriz", conjunto)
        for k, nombre in ETIQUETAS.items():
            f = os.path.join(ruta, f"_pred_{k}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            filas.append({"Conjunto": conjunto, "Modelo": nombre,
                          **{m: round(v, 4) for m, v in
                             metricas(d["y"].values, d["p"].values).items()}})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(MODELOS, "tabla_5_19_matriz.csv"), index=False)
    print(t.to_string(index=False))
    return t


# ---------------------------------------------------------------------------
def fase_tabla():
    """Consolida V0, V1 y V2 en una sola tabla comparable."""
    filas = []
    for var, ruta in (("V0", rutas.artefacto("")),
                      ("V1", rutas.artefacto("v1")),
                      ("V2", rutas.artefacto("v2"))):
        for k, nombre in ETIQUETAS.items():
            f = os.path.join(ruta, f"_pred_{k}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            filas.append({"Variante": var, "Modelo": nombre,
                          **{m: round(v, 4) for m, v in
                             metricas(d["y"].values, d["p"].values).items()}})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(MODELOS, "tabla_5_18_variantes.csv"), index=False)
    print(t.to_string(index=False))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", required=True,
                    choices=["fuga", "v1", "v2", "matriz", "matriz-tabla",
                             "tabla"])
    ap.add_argument("--conjunto", choices=["propias", "propagadas"],
                    default="propias")
    ap.add_argument("--solo", choices=list(ETIQUETAS), default=None)
    ap.add_argument("--rehacer", action="store_true")
    a = ap.parse_args()

    if a.fase == "tabla":
        fase_tabla()
        return
    if a.fase == "matriz-tabla":
        fase_matriz_tabla()
        return

    df, aristas = cargar()
    if a.fase == "fuga":
        fase_fuga(df)
    elif a.fase == "matriz":
        fase_matriz(df, aristas, a.conjunto, a.solo, a.rehacer)
    else:
        fase_comparativa(df, aristas, a.fase, a.solo, a.rehacer)


if __name__ == "__main__":
    main()
