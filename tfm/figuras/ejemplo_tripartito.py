#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figura didáctica de la estructura tripartita entidad–UT–socio.

Dibuja, sobre tres entidades reales del universo construido, los tres tipos de
vínculo que forman el triángulo de riesgo: la adjudicación de la entidad a una
unión temporal, la integración de esa unión por sus socios y la contratación
directa de un socio con la misma entidad, que es la arista que cierra el ciclo.

Los tres casos se eligen por criterio y no a mano: entre las entidades que
cierran exactamente un triángulo —las de estructura más legible, porque no hay
aristas que se superpongan— se toman las tres primeras en orden de
identificador, que es un criterio reproducible y ajeno a lo que se quiere
mostrar. Los actores aparecen bajo seudónimo.

La definición del triángulo no se repite aquí: se importa de
`tfm.grafo.triangulos`, que es donde vive.

Salida: OUTPUTS/figuras/fig_tripartita_ejemplo.pdf

Uso:
    python -m tfm.figuras.ejemplo_tripartito
"""

import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tfm import rutas
from tfm.grafo import triangulos as tri

PQ = rutas.parquet()
SALIDA = rutas.figuras_publicadas()

AZUL_ENTIDAD = "#1F4E79"
NARANJA = "#D95F02"
VERDE = "#1B9E77"
ROJO = "#C1272D"
GRIS = "#58595B"

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "figure.dpi": 200,
})

def triangulos():
    """Los triángulos entidad–UT–socio del universo y el mapa de seudónimos.

    La exclusión de banca y seguros y la condición de que los tres vértices
    sean actores distintos vienen de `tfm.grafo.triangulos`, que es la
    definición que emplean también el brazo clásico, el solapamiento y las
    figuras de red.
    """
    con = duckdb.connect()
    t = tri.triangulos(con, PQ)
    seudo = con.execute(
        f"SELECT id_nodo, seudonimo FROM "
        f"'{PQ}/nodo_seudonimo/nodo_seudonimo.parquet'").df()
    con.close()
    return t, dict(zip(seudo["id_nodo"], seudo["seudonimo"]))


def elegir(t, cuantas=3):
    """Tres triángulos de lectura clara: una entidad, una UT y un socio.

    Se toman las entidades que cierran un único triángulo, que son las que
    caben en una figura sin que las aristas se superpongan. Las ternas ya
    llegan con tres actores distintos, porque esa condición forma parte de la
    definición del triángulo.
    """
    propios = t
    conteo = propios.groupby("entidad").size()
    candidatas = sorted(conteo[conteo == 1].index)[:cuantas]
    return propios[propios["entidad"].isin(candidatas)].reset_index(drop=True)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    t, seudo = triangulos()
    sel = elegir(t)
    print(f"{len(t):,} triángulos en el universo; se dibujan {len(sel)}"
          .replace(",", "."))

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ancho = 4.0
    for k, fila in sel.iterrows():
        x = k * ancho
        pos = {"entidad": (x + 1.0, 2.0), "ut": (x + 0.0, 0.4),
               "socio": (x + 2.0, 0.4)}
        #: adjudica: entidad -> UT
        ax.plot(*zip(pos["entidad"], pos["ut"]), color=AZUL_ENTIDAD, lw=1.4,
                zorder=1)
        #: integra: UT -> socio
        ax.plot(*zip(pos["ut"], pos["socio"]), color=NARANJA, lw=1.4, zorder=1)
        #: contrata directo: socio -> entidad, la arista que cierra el ciclo
        ax.plot(*zip(pos["socio"], pos["entidad"]), color=ROJO, lw=1.4,
                ls="--", zorder=1)
        for tipo, color in (("entidad", AZUL_ENTIDAD), ("ut", NARANJA),
                            ("socio", VERDE)):
            xx, yy = pos[tipo]
            ax.scatter([xx], [yy], s=340, color=color, zorder=2,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(seudo.get(fila[tipo], "—"), (xx, yy),
                        xytext=(0, -14 if tipo != "entidad" else 12),
                        textcoords="offset points", ha="center",
                        fontsize=6.5, color=GRIS)

    ax.set_xlim(-1.2, ancho * len(sel) - 1.2)
    ax.set_ylim(-0.8, 2.9)
    ax.axis("off")
    ax.legend(handles=[
        Line2D([], [], color=AZUL_ENTIDAD, lw=1.4, label="adjudica (entidad → UT)"),
        Line2D([], [], color=NARANJA, lw=1.4, label="integra (UT → socio)"),
        Line2D([], [], color=ROJO, lw=1.4, ls="--",
               label="contrata directo (socio → entidad)"),
    ], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))

    ruta = os.path.join(SALIDA, "fig_tripartita_ejemplo.pdf")
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"-> {ruta}")


if __name__ == "__main__":
    main()
