#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contraste entre la GCN y el modelo relacional que la simplifica.

El apartado 5.2.8 compara las dos arquitecturas métrica a métrica. Sus
diferencias son cifras publicadas del trabajo, de modo que se calculan aquí y
se guardan como artefacto en lugar de restarse en el texto: cada número del
documento tiene que proceder de una celda.

Toma las métricas de la GCN de `gnn/metricas_gcn.json` y las del modelo
relacional de la tabla de los nueve ordenamientos, que es donde ambos se miden
sobre el mismo conjunto de prueba.

Salida: tabla_5_31_gcn_vs_sgc.csv

Uso:
    python -m tfm.modelos.gcn_vs_sgc
"""

import csv
import json
import os

from tfm import rutas

#: La fila del modelo relacional en la tabla de los nueve ordenamientos.
FILA_SGC = "SGC - propagación + MLP"

MÉTRICAS = ("AUC-ROC", "PR-AUC", "F1", "Precision@200", "Recall@200")


def main():
    with open(rutas.artefacto(os.path.join("gnn", "metricas_gcn.json")),
              encoding="utf-8") as fh:
        gcn = json.load(fh)

    sgc = None
    with open(rutas.artefacto("tabla_5_29_nueve_ordenamientos.csv"),
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["Ordenamiento"].strip() == FILA_SGC:
                sgc = fila
    if sgc is None:
        raise SystemExit(f"no se encuentra la fila «{FILA_SGC}» en la tabla "
                         "de los nueve ordenamientos")

    filas = []
    for m in MÉTRICAS:
        a, b = float(gcn[m]), float(sgc[m])
        filas.append({"Métrica": m, "GCN de dos capas": f"{a:.4f}",
                      "SGC - propagación + MLP": f"{b:.4f}",
                      "Diferencia": f"{b - a:.4f}"})

    destino = os.path.join(rutas.resultados(), "tabla_5_31_gcn_vs_sgc.csv")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    for f in filas:
        print(f"  {f['Métrica']:<14} {f['GCN de dos capas']:>8} "
              f"{f['SGC - propagación + MLP']:>8} {f['Diferencia']:>9}")
    print(f"\n-> {destino}")
    convergencia(gcn)


#: Hitos de la trayectoria de validación que el apartado 5.2.7 cita. Se
#: publican como tabla y no se leen del histórico completo: una serie de
#: ciento cincuenta AUC daría respaldo por casualidad a casi cualquier cifra
#: de cuatro decimales, y el control de cifras dejaría de comprobar nada.
HITOS = (1200, 2000, 4000, 6000)


def convergencia(gcn):
    """La trayectoria de validación de la GCN en unos pocos hitos declarados."""
    historia = {}
    with open(rutas.artefacto(os.path.join("gnn", "historia_gcn.csv")),
              encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            historia[int(fila["epoca"])] = float(fila["auc_validacion"])

    filas = [{"Época": e, "AUC-ROC de validación": f"{historia[e]:.4f}",
              "Hito": "final de la rejilla" if e == 1200 else "ajuste largo"}
             for e in HITOS if e in historia]
    filas.append({"Época": gcn["mejor_epoca"],
                  "AUC-ROC de validación": f"{float(gcn['auc_validacion']):.4f}",
                  "Hito": "máximo de validación"})
    filas.append({"Época": gcn["epocas_recorridas"],
                  "AUC-ROC de validación":
                      f"{historia[gcn['epocas_recorridas']]:.4f}",
                  "Hito": "parada temprana"})

    destino = os.path.join(rutas.resultados(),
                           "tabla_5_32_convergencia_gcn.csv")
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for f in filas:
        print(f"  época {f['Época']:>6}   {f['AUC-ROC de validación']}   "
              f"{f['Hito']}")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
