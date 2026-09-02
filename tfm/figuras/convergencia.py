#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de convergencia de las dos redes con paso de mensajes.

Muestra el AUC de validación frente a la época para las cuatro combinaciones de
la rejilla y para el ajuste largo de la configuración ganadora, en la GCN y en
GraphSAGE. Es la evidencia de que la cifra publicada de cada arquitectura
corresponde a un modelo convergido y no al final del presupuesto.

Salida: OUTPUTS/figuras/fig_convergencia_gnn.pdf

Uso:
    python -m tfm.figuras.convergencia
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tfm import rutas

SALIDA = rutas.figuras_publicadas()

AZUL = "#0097CC"
GRIS = "#58595B"
ACENTO = "#D95F02"

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 200,
})

TITULO = {"gcn": "GCN de dos capas", "sage": "GraphSAGE con muestreo"}


def artefacto(modo, nombre):
    return rutas.artefacto(os.path.join("gnn", nombre.format(modo=modo)))


def main():
    os.makedirs(SALIDA, exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)

    for ax, modo in zip(ejes, ("gcn", "sage")):
        with open(artefacto(modo, "metricas_{modo}.json"), encoding="utf-8") as fh:
            met = json.load(fh)
        #: Las cuatro combinaciones de la rejilla, en gris: sirven para situar
        #: la elegida, no para leerlas una a una.
        for oculta in (16, 32):
            for lr in (0.01, 0.03):
                ruta = artefacto(modo, f"historia_{{modo}}_{oculta}_{lr}.csv")
                if not os.path.exists(ruta):
                    continue
                h = pd.read_csv(ruta)
                ax.plot(h["epoca"], h["auc_validacion"], color=GRIS, lw=0.7,
                        alpha=0.55)
        larga = pd.read_csv(artefacto(modo, "historia_{modo}.csv"))
        ax.plot(larga["epoca"], larga["auc_validacion"], color=AZUL, lw=1.3,
                label=f"configuración elegida: {met['oculta']} unidades, "
                      f"tasa {met['lr']}")
        ax.axvline(met["mejor_epoca"], color=ACENTO, lw=0.9, ls="--")
        ax.annotate(f"mejor validación\népoca {met['mejor_epoca']:,}"
                    .replace(",", "."),
                    xy=(met["mejor_epoca"], met["auc_validacion"]),
                    xytext=(6, -16), textcoords="offset points",
                    color=ACENTO, fontsize=6)
        ax.set_title(TITULO[modo])
        ax.set_xlabel("épocas")
        ax.legend(loc="lower right")

    ejes[0].set_ylabel("AUC-ROC de validación")
    fig.tight_layout()
    destino = os.path.join(SALIDA, "fig_convergencia_gnn.pdf")
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
