#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intervalos de confianza de las métricas, por remuestreo.

El Capítulo 5 compara los ordenamientos sobre una única partición y con una
única semilla, y de ahí sale el ganador. Sin una medida de dispersión, una
diferencia de seis milésimas de AUC-ROC entre dos modelos no se distingue del
ruido de esa partición, y esa es la primera objeción que admite la comparación.

Este módulo la responde por la vía habitual cuando no se puede reentrenar:
**remuestreo con reposición del conjunto de prueba**. Las predicciones están
fijadas —los modelos no se reajustan—, de modo que lo que se mide es la
variabilidad que introduce la muestra de evaluación, que es exactamente la
fuente de incertidumbre que la comparación ignora.

## Qué produce

1. Un intervalo de percentiles al 95 % para el AUC-ROC de cada ordenamiento.
2. Para cada pareja que el texto compara, el intervalo de la **diferencia
   emparejada** —calculada sobre el mismo remuestreo, que es lo que cancela la
   variabilidad común— y la fracción de remuestreos en que la diferencia
   conserva el signo. Esa fracción es la que dice si el orden entre dos modelos
   es afirmable.

## Lo que no mide

La incertidumbre del **ajuste**: haría falta reentrenar con particiones
distintas, y la GCN sola son horas. Lo que aquí se acota es la incertidumbre de
la **evaluación**, que es la que afecta a la comparación entre modelos ya
entrenados sobre el mismo conjunto.

Salida: tablas/tabla_5_34_incertidumbre.csv y modelos/incertidumbre.json

Uso:
    python -m tfm.modelos.incertidumbre
    python -m tfm.modelos.incertidumbre --remuestreos 5000
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tfm import rutas

SEMILLA = 42
MODELOS = rutas.resultados()

#: De dónde sale la puntuación de cada ordenamiento. Son los mismos artefactos
#: que alimentan la tabla de los nueve ordenamientos, para que los intervalos
#: describan las cifras publicadas y no otras.
FUENTES = [
    ("SNA — Regresión Logística", os.path.join("v1", "_pred_lr.csv")),
    ("ML — Random Forest", os.path.join("v1", "_pred_rf.csv")),
    ("ML — Gradient Boosting", os.path.join("v1", "_pred_gb.csv")),
    ("SGC — propagación + MLP", os.path.join("v1", "_pred_gnn.csv")),
    ("GNN — GCN de dos capas", os.path.join("gnn", "_pred_gcn.csv")),
    ("GNN — GraphSAGE con muestreo", os.path.join("gnn", "_pred_sage.csv")),
    ("ML — Random Forest sobre variables propagadas",
     os.path.join("matriz", "propagadas", "_pred_rf.csv")),
]

#: Las comparaciones que el texto convierte en conclusión. Cada una se contrasta
#: emparejada: sobre el mismo remuestreo, no sobre dos independientes.
PAREJAS = [
    ("ML — Random Forest sobre variables propagadas",
     "GNN — GraphSAGE con muestreo"),
    ("GNN — GraphSAGE con muestreo", "SGC — propagación + MLP"),
    ("SGC — propagación + MLP", "GNN — GCN de dos capas"),
    ("SGC — propagación + MLP", "ML — Random Forest"),
]


def cargar():
    """Las puntuaciones de todos los ordenamientos, alineadas fila a fila.

    Los artefactos vienen de módulos distintos, así que no basta con suponer
    que comparten el orden: se comprueba que el vector de etiquetas es idéntico
    en todos. Si no lo fuera, la diferencia emparejada compararía actores
    distintos y el intervalo no significaría nada.
    """
    y_ref, puntuaciones = None, {}
    for nombre, rel in FUENTES:
        ruta = rutas.artefacto(rel)
        if not os.path.exists(ruta):
            print(f"  [!] falta {rel}; se omite {nombre}")
            continue
        d = pd.read_csv(ruta)
        y = d["y"].values.astype(int)
        if y_ref is None:
            y_ref = y
        elif len(y) != len(y_ref) or not np.array_equal(y, y_ref):
            raise SystemExit(
                f"{rel} no está alineado con el resto del conjunto de prueba: "
                "la comparación emparejada exige el mismo orden de filas")
        puntuaciones[nombre] = d["p"].values.astype(float)
    if y_ref is None:
        raise SystemExit("no hay ninguna predicción que leer")
    return y_ref, puntuaciones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remuestreos", type=int, default=2000)
    a = ap.parse_args()

    y, punt = cargar()
    n = len(y)
    print(f"conjunto de prueba: {n:,} actores, {int(y.sum()):,} positivos"
          .replace(",", "."))
    print(f"{len(punt)} ordenamientos · {a.remuestreos} remuestreos")

    rng = np.random.default_rng(SEMILLA)
    #: Los índices se generan una vez y se comparten: es lo que hace que la
    #: diferencia entre dos modelos se mida sobre la misma muestra.
    indices = rng.integers(0, n, size=(a.remuestreos, n))

    aucs = {}
    for nombre, p in punt.items():
        v = np.empty(a.remuestreos)
        for b in range(a.remuestreos):
            i = indices[b]
            yb = y[i]
            #: Un remuestreo sin positivos o sin negativos no define el AUC.
            v[b] = roc_auc_score(yb, p[i]) if 0 < yb.sum() < len(yb) else np.nan
        aucs[nombre] = v
        print(f"  {nombre:46} {np.nanmean(v):.4f}")

    filas = []
    for nombre, p in punt.items():
        v = aucs[nombre]
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        filas.append({"Ordenamiento": nombre,
                      "AUC-ROC": round(float(roc_auc_score(y, p)), 4),
                      "IC 95% inferior": round(float(lo), 4),
                      "IC 95% superior": round(float(hi), 4),
                      "Amplitud": round(float(hi - lo), 4)})
    tabla = pd.DataFrame(filas).sort_values("AUC-ROC", ascending=False)

    comparaciones = []
    for a1, a2 in PAREJAS:
        if a1 not in aucs or a2 not in aucs:
            continue
        d = aucs[a1] - aucs[a2]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        conserva = float(np.nanmean(d > 0))
        comparaciones.append({
            "Comparación": f"{a1} − {a2}",
            "Diferencia observada": round(
                float(roc_auc_score(y, punt[a1]) - roc_auc_score(y, punt[a2])), 4),
            "IC 95% inferior": round(float(lo), 4),
            "IC 95% superior": round(float(hi), 4),
            "Remuestreos a favor": round(conserva, 4),
            "Afirmable": "sí" if lo > 0 or hi < 0 else "no"})
    comp = pd.DataFrame(comparaciones)

    os.makedirs(MODELOS, exist_ok=True)
    tabla.to_csv(os.path.join(MODELOS, "tabla_5_34_incertidumbre.csv"),
                 index=False)
    comp.to_csv(os.path.join(MODELOS, "tabla_5_35_comparaciones.csv"),
                index=False)
    with open(os.path.join(MODELOS, "incertidumbre.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"remuestreos": a.remuestreos, "semilla": SEMILLA,
                   "n_prueba": int(n), "positivos": int(y.sum()),
                   "amplitud_media_ic": round(float(tabla["Amplitud"].mean()), 4),
                   "comparaciones_afirmables": int((comp["Afirmable"] == "sí").sum()),
                   "comparaciones": len(comp)}, fh, ensure_ascii=False, indent=2)

    print("\n" + tabla.to_string(index=False))
    print("\n" + comp.to_string(index=False))
    print(f"\n-> {MODELOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
