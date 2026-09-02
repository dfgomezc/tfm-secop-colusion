#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura de la explicación de las alertas (OE-6).

Dos paneles sobre los artefactos de `tfm.modelos.explicabilidad`, referidos por
defecto al modelo relacional del trabajo —la propagación al estilo SGC seguida
del perceptrón—, que es el que la explicación abre:

* el reparto entre el nodo y su entorno en las cien alertas de mayor
  puntuación, ordenadas por la parte que procede del entorno;
* el desglose por variable de una alerta concreta, con el color indicando de
  qué bloque de propagación procede cada aportación.

Salida: OUTPUTS/figuras/fig_explicabilidad.pdf

Uso:
    python -m tfm.figuras.explicabilidad
    python -m tfm.figuras.explicabilidad --modelo rf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tfm import rutas

SALIDA = rutas.figuras_publicadas()

AZUL = "#0097CC"      # color primario de las figuras
GRIS = "#58595B"
ACENTO = "#D95F02"

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 200,
})

#: Un color por bloque de propagación. El del nodo propio es el corporativo; el
#: entorno se aleja de él a medida que se aleja en saltos.
COLOR = {"propio": AZUL, "un salto": ACENTO, "dos saltos": GRIS}

LEGIBLE = {
    "strength": "grado ponderado",
    "pagerank": "PageRank",
    "en_componente_gigante": "en comp. gigante",
    "log_valor_total": "log valor total",
    "log_valor_medio": "log valor medio",
    "n_procesos": "n.º de procesos",
    "n_cuentas_plataforma": "cuentas en plataforma",
}


#: Cómo se nombra cada modelo en el título del panel derecho.
TITULO = {"mlp": "modelo relacional", "rf": "Random Forest"}


def artefacto(modelo, nombre):
    return rutas.artefacto(os.path.join("explicabilidad", modelo, nombre))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=("mlp", "rf"), default="mlp")
    a = ap.parse_args()

    os.makedirs(SALIDA, exist_ok=True)
    bloques = pd.read_csv(artefacto(a.modelo, "atribucion_bloques.csv"))
    variables = pd.read_csv(artefacto(a.modelo, "atribucion_variables.csv"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # -- panel izquierdo: reparto en las cien alertas -----------------------
    b = bloques.copy()
    b["total"] = b[["propio", "un_salto", "dos_saltos"]].abs().sum(axis=1)
    for c in ("propio", "un_salto", "dos_saltos"):
        b[c + "_pct"] = 100 * b[c].abs() / b["total"]
    b = b.sort_values("propio_pct").reset_index(drop=True)
    x = np.arange(len(b))
    a1.bar(x, b["propio_pct"], width=1.0, color=COLOR["propio"],
           label="del propio actor")
    a1.bar(x, b["un_salto_pct"], width=1.0, bottom=b["propio_pct"],
           color=COLOR["un salto"], label="de su entorno a un salto")
    a1.bar(x, b["dos_saltos_pct"], width=1.0,
           bottom=b["propio_pct"] + b["un_salto_pct"],
           color=COLOR["dos saltos"], label="de su entorno a dos saltos")
    a1.axhline(50, color="white", lw=0.8, ls="--")
    a1.set_xlim(-0.5, len(b) - 0.5)
    a1.set_ylim(0, 100)
    a1.set_xlabel("las cien alertas de mayor puntuación, ordenadas")
    a1.set_ylabel("reparto de la atribución (%)")
    a1.set_title("De dónde procede el riesgo que se imputa")
    a1.legend(loc="lower right", framealpha=0.9)
    a1.grid(False)

    # -- panel derecho: una alerta concreta --------------------------------
    caso = bloques.iloc[0]["seudonimo"]
    v = variables[variables["seudonimo"] == caso].copy()
    v["abs"] = v["atribucion"].abs()
    v = v.sort_values("abs", ascending=False).head(9).iloc[::-1]
    etiquetas = [f"{LEGIBLE.get(r.variable, r.variable)}\n({r.bloque})"
                 for r in v.itertuples()]
    a2.barh(np.arange(len(v)), v["atribucion"],
            color=[COLOR[b] for b in v["bloque"]])
    a2.set_yticks(np.arange(len(v)), etiquetas)
    a2.axvline(0, color=GRIS, lw=0.8)
    a2.set_xlabel("aportación a la puntuación")
    a2.set_title(f"Desglose de la alerta {caso} ({TITULO[a.modelo]})")

    fig.tight_layout()
    sufijo = "" if a.modelo == "mlp" else f"_{a.modelo}"
    destino = os.path.join(SALIDA, f"fig_explicabilidad{sufijo}.pdf")
    fig.savefig(destino, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
