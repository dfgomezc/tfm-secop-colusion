#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Métricas de lista corta, umbral y calibración.

El documento reportaba el Recall@200 sin declarar su techo aritmético, lo que
invertía la lectura del resultado: con 2.516 positivos en el conjunto de prueba,
ninguna lista de 200 casos puede recuperar más del 7,95 % de ellos.

Este script produce lo que un organismo de control necesita para decidir:

  - Precision@k y Lift@k para varios tamaños de lista.
  - El F1 al umbral 0,5 y al umbral que lo maximiza en validación.
  - Matriz de confusión al umbral declarado.
  - Calibración: puntuación de Brier y diagrama de fiabilidad.

No reentrena nada: parte de las predicciones ya persistidas.

Salidas en OUTPUTS/:
    tabla_5_25_precision_k.csv     Precision@k y Lift@k por ordenamiento
    tabla_5_26_umbral_f1.csv       F1 a 0,5 y al umbral óptimo
    tabla_5_27_confusion.csv       matriz de confusión del mejor modelo
    tabla_5_28_calibracion.csv     Brier y diagrama de fiabilidad
    curva_lift.csv                 para la figura

Uso:
    python -m tfm.modelos.metricas_operativas
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score

from tfm import rutas

SEMILLA = 42
KS = (50, 100, 200, 500, 1000)
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
MODELOS = rutas.resultados()

#: Los cuatro modelos de la variante sin fuga, más los tres ordenamientos
#: clásicos. Se comparan sobre el mismo conjunto de prueba.
ENTRENADOS = {"lr": "SNA — Regresión Logística", "rf": "ML — Random Forest",
              "gb": "ML — Gradient Boosting", "gnn": "SGC — propagación + MLP"}


def en_k(y, p, k, rng):
    """Precisión, exhaustividad y lift sobre los k primeros."""
    orden = np.lexsort((rng.random(len(p)), -p))[:k]
    prec = float(y[orden].mean())
    return prec, float(y[orden].sum() / y.sum()), prec / float(y.mean())


def cargar():
    """Predicciones de los cuatro modelos y de los tres ordenamientos clásicos."""
    salida = {}
    for k, nombre in ENTRENADOS.items():
        f = rutas.artefacto(os.path.join("v1", f"_pred_{k}.csv"))
        if os.path.exists(f):
            d = pd.read_csv(f)
            salida[nombre] = (d["y"].values, d["p"].values)

    #: Los clásicos se reconstruyen desde la partición persistida, que ya lleva
    #: el id_nodo de cada fila del conjunto de prueba.
    part = rutas.artefacto("particion_prueba.csv")
    band = rutas.artefacto("banderas_rojas.csv")
    if os.path.exists(part):
        import duckdb
        con = duckdb.connect()
        ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
        fx = con.execute(f"SELECT id_nodo, pagerank, n_contratos "
                         f"FROM read_parquet('{ruta}')").df()
        pr = pd.read_csv(part)
        d = pr.merge(fx, on="id_nodo", how="left")
        if os.path.exists(band):
            d = d.merge(pd.read_csv(band)[["id_nodo", "banderas_rojas"]],
                        on="id_nodo", how="left")
            salida["Clásico — Banderas rojas"] = (
                d["riesgo_alto"].values, d["banderas_rojas"].fillna(0).values)
        salida["Clásico — PageRank puro"] = (d["riesgo_alto"].values,
                                             d["pagerank"].values)
        salida["Trivial — Número de contratos"] = (d["riesgo_alto"].values,
                                                   d["n_contratos"].values)
        con.close()
    return salida


def main():
    datos = cargar()
    if not datos:
        raise SystemExit("faltan las predicciones; ejecuta antes 10 y 11")

    y0 = next(iter(datos.values()))[0]
    prev = float(y0.mean())
    n_pos = int(y0.sum())
    print(f"conjunto de prueba: {len(y0):,} filas · {n_pos:,} positivos "
          f"· prevalencia {prev:.4f}")
    print(f"techo del Recall@200: 200/{n_pos} = {200/n_pos:.4f}\n")

    # ---- Precision@k y Lift@k ------------------------------------------
    filas, curva = [], []
    for nombre, (y, p) in datos.items():
        rng = np.random.default_rng(SEMILLA)
        fila = {"Ordenamiento": nombre}
        for k in KS:
            prec, rec, lift = en_k(y, p, k, rng)
            fila[f"P@{k}"] = round(prec, 4)
            fila[f"Lift@{k}"] = round(lift, 2)
            curva.append({"modelo": nombre, "k": k, "precision": prec,
                          "recall": rec, "lift": lift})
        filas.append(fila)
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(MODELOS, "tabla_5_25_precision_k.csv"), index=False)
    pd.DataFrame(curva).to_csv(os.path.join(MODELOS, "curva_lift.csv"),
                               index=False)
    print(t.to_string(index=False))

    # ---- F1 al umbral 0,5 y al umbral óptimo ----------------------------
    filas = []
    for nombre, (y, p) in datos.items():
        if p.max() > 1.0 or p.min() < 0.0:
            continue  # los ordenamientos clásicos no producen probabilidades
        f_05 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
        #: El umbral óptimo se busca sobre una rejilla fina de los propios
        #: valores predichos, no sobre una malla arbitraria.
        cand = np.quantile(p, np.linspace(0.50, 0.999, 200))
        f1s = [f1_score(y, (p >= u).astype(int), zero_division=0) for u in cand]
        i = int(np.argmax(f1s))
        filas.append({"Modelo": nombre, "F1 al umbral 0,5": round(f_05, 4),
                      "Umbral óptimo": round(float(cand[i]), 4),
                      "F1 al umbral óptimo": round(float(f1s[i]), 4),
                      "Señalados al óptimo": int((p >= cand[i]).sum())})
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(MODELOS, "tabla_5_26_umbral_f1.csv"), index=False)
    print("\n" + t.to_string(index=False))

    # ---- Matriz de confusión y calibración del mejor modelo -------------
    mejor = "SGC — propagación + MLP"
    y, p = datos[mejor]
    filas = []
    for etiqueta, u in (("0,5", 0.5),
                        ("óptimo de F1", float(
                            t.set_index("Modelo").loc[mejor, "Umbral óptimo"]))):
        tn, fp, fn, tp = confusion_matrix(y, (p >= u).astype(int)).ravel()
        filas.append({"Umbral": etiqueta, "Verdaderos positivos": tp,
                      "Falsos positivos": fp, "Falsos negativos": fn,
                      "Verdaderos negativos": tn,
                      "Precisión": round(tp / max(tp + fp, 1), 4),
                      "Exhaustividad": round(tp / max(tp + fn, 1), 4)})
    c = pd.DataFrame(filas)
    c.to_csv(os.path.join(MODELOS, "tabla_5_27_confusion.csv"), index=False)
    print("\n" + c.to_string(index=False))

    #: Calibración: ¿una probabilidad de 0,8 corresponde al 80 % de aciertos?
    filas = []
    for nombre, (yy, pp) in datos.items():
        if pp.max() > 1.0 or pp.min() < 0.0:
            continue
        filas.append({"Modelo": nombre,
                      "Brier": round(brier_score_loss(yy, pp), 4)})
    b = pd.DataFrame(filas)

    bordes = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bordes) - 1, 0, 9)
    fiab = []
    for i in range(10):
        m = idx == i
        if m.sum() == 0:
            continue
        fiab.append({"tramo": f"[{bordes[i]:.1f}, {bordes[i+1]:.1f})",
                     "n": int(m.sum()),
                     "probabilidad media": round(float(p[m].mean()), 4),
                     "frecuencia observada": round(float(y[m].mean()), 4)})
    f = pd.DataFrame(fiab)
    pd.concat([b.assign(tramo="", n="", **{"probabilidad media": "",
                                           "frecuencia observada": ""}),
               f.assign(Modelo="", Brier="")]).to_csv(
        os.path.join(MODELOS, "tabla_5_28_calibracion.csv"), index=False)
    print("\n" + b.to_string(index=False))
    print("\ndiagrama de fiabilidad del mejor modelo")
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
