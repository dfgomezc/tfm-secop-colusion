#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figuras de distribuciones y métricas del Capítulo 5.

Todas se dibujan desde las tablas publicadas de `OUTPUTS/`, de modo que las
cifras del texto y las de las figuras proceden de la misma fuente y no pueden
divergir.

Salida: OUTPUTS/figuras/*.pdf, en vectorial.

Uso:
    python -m tfm.figuras.generales
"""

import os
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
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


def miles(x, _):
    return f"{int(x):,}".replace(",", ".")


def compacto(x, _):
    """Rótulos de eje abreviados: a media anchura de página los separadores
    de millar hacen que las marcas se solapen entre sí."""
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f} M".replace(".", ",")
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f} k"
    return f"{int(x)}"


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {nombre}")


def panel_embudo(con, ax):
    """Cuántos contratos sobreviven a cada regla del alcance."""
    total = con.execute("""
        SELECT count(*) FROM fact_contrato
        WHERE fecha_firma BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
    """).fetchone()[0]
    alcance = con.execute("SELECT count(*) FROM contrato_2025").fetchone()[0]
    rescatados = con.execute("""
        SELECT count(*) FROM contrato_2025
        WHERE tipo_de_contrato = 'Prestación de servicios'
          AND proveedor_es_persona_natural""").fetchone()[0]

    #: Etiquetas cortas: con el pie completo los tres rótulos se solapaban al
    #: reducir la figura a media anchura de página.
    etiquetas = ["Firmados\nen 2025", "Sin prestación de\nservicios de natural",
                 "Más el núcleo\nde red"]
    valores = [total, alcance - rescatados, alcance]

    barras = ax.bar(etiquetas, valores, color=[GRIS, AZUL, AZUL], width=0.55)
    barras[2].set_color(ACENTO)
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,}".replace(",", "."),
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_ylabel("Contratos")
    ax.yaxis.set_major_formatter(FuncFormatter(compacto))
    ax.set_ylim(0, total * 1.15)


def panel_nodos(con, ax):
    """Composición del universo de nodos y cobertura del registro mercantil."""
    df = con.execute("""
        SELECT tipo_nodo,
               count(*) AS nodos,
               count(*) FILTER (WHERE cruza_rues) AS con_rues
        FROM nodo_2025 GROUP BY 1 ORDER BY 2 DESC""").df()
    NOMBRE = {"persona_juridica": "Persona jurídica",
              "union_temporal": "Unión temporal",
              "persona_natural": "Persona natural",
              "sin_clasificar": "Sin clasificar"}
    etq = [NOMBRE.get(t, t.replace("_", " ").capitalize())
           for t in df["tipo_nodo"]]

    y = range(len(df))
    ax.barh(y, df["nodos"], color=AZUL, label="Nodos del universo")
    ax.barh(y, df["con_rues"], color=GRIS, label="Cruzan con el RUES")
    ax.set_yticks(list(y))
    ax.set_yticklabels(etq)
    ax.invert_yaxis()
    ax.set_xlabel("Actores económicos")
    ax.xaxis.set_major_formatter(FuncFormatter(compacto))
    for i, (n, r) in enumerate(zip(df["nodos"], df["con_rues"])):
        ax.text(n * 1.01, i, f"{n:,}".replace(",", "."), va="center", fontsize=6)
    ax.legend(frameon=False, loc="lower right")


def panel_vinculos(con, ax):
    """Peso relativo de cada tipo de arista."""
    df = con.execute("""
        SELECT tipo_vinculo, count(*) AS aristas
        FROM vinculo_2025 GROUP BY 1 ORDER BY 2 DESC""").df()
    NOMBRE = {"co_presentacion": "Co-presentación", "se_presenta_a": "Se presenta a",
              "co_oferta": "Co-oferta", "adjudica": "Adjudica",
              "comparte_contacto": "Comparte contacto", "integra": "Integra UT",
              "representado_por": "Representado por"}
    etq = [NOMBRE.get(t, t.replace("_", " ")) for t in df["tipo_vinculo"]]

    barras = ax.barh(range(len(df)), df["aristas"], color=AZUL)
    barras[0].set_color(GRIS)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(etq)
    ax.invert_yaxis()
    ax.set_xlabel("Aristas")
    ax.xaxis.set_major_formatter(FuncFormatter(compacto))
    for i, v in enumerate(df["aristas"]):
        ax.text(v * 1.01, i, f"{v:,}".replace(",", "."), va="center", fontsize=6)


def panel_competencia(con, ax):
    """Distribución del número de oferentes por proceso: el peso del
    oferente único y la cola de los acuerdos marco."""
    df = con.execute("""
        SELECT CASE WHEN n_oferentes = 0 THEN '0'
                    WHEN n_oferentes = 1 THEN '1'
                    WHEN n_oferentes BETWEEN 2 AND 5   THEN '2-5'
                    WHEN n_oferentes BETWEEN 6 AND 20  THEN '6-20'
                    WHEN n_oferentes BETWEEN 21 AND 100 THEN '21-100'
                    ELSE '>100' END AS tramo,
               count(*) AS procesos
        FROM proceso_competencia GROUP BY 1""").df()
    orden = ["0", "1", "2-5", "6-20", "21-100", ">100"]
    df = df.set_index("tramo").reindex(orden).fillna(0).reset_index()

    colores = [GRIS, ACENTO, AZUL, AZUL, AZUL, GRIS]
    barras = ax.bar(df["tramo"], df["procesos"], color=colores, width=0.6)
    for b, v in zip(barras, df["procesos"]):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                    f"{int(v):,}".replace(",", "."), ha="center", va="bottom", fontsize=6)
    ax.set_xlabel("Oferentes por proceso")
    ax.set_ylabel("Procesos")
    ax.tick_params(axis="x", labelsize=6)
    ax.yaxis.set_major_formatter(FuncFormatter(compacto))
    ax.set_ylim(0, df["procesos"].max() * 1.15)


def panel_splink(con, ax):
    """Peso en bits de cada señal aprendida por el modelo de identidad."""
    try:
        df = con.execute("""
            SELECT comparacion || ' — ' || nivel AS senal,
                   log2(m_probability / nullif(u_probability, 0)) AS bits
            FROM splink_parametros
            WHERE m_probability IS NOT NULL AND u_probability > 0
              AND log2(m_probability / nullif(u_probability, 0)) > 3
            ORDER BY bits DESC LIMIT 9""").df()
    except Exception as exc:
        print(f"  [!] sin splink_parametros: {str(exc)[:60]}")
        return False
    if df.empty:
        print("  [!] splink_parametros sin filas utilizables")
        return False

    etq = [s.replace("Exact match on ", "").replace("_", " ")[:44]
           for s in df["senal"]]
    ax.barh(range(len(df)), df["bits"], color=AZUL)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(etq)
    ax.invert_yaxis()
    ax.set_xlabel("Evidencia a favor de la coincidencia (bits)")
    for i, v in enumerate(df["bits"]):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=6)


TABLAS = ["fact_contrato", "contrato_2025", "nodo_2025", "vinculo_2025",
          "proceso_competencia", "splink_parametros"]


def abrir():
    """Vistas sobre OUTPUTS/gold en vez de la base .duckdb.

    Las bases .duckdb guardan la ruta absoluta de los Parquet, así que dejan de
    funcionar si el proyecto se abre desde otra máquina o desde otra unidad.
    Leyendo la carpeta directamente el script funciona en cualquier equipo.
    """
    pq = rutas.parquet()
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    for t in TABLAS:
        patron = os.path.join(pq, t, "*.parquet").replace("\\", "/")
        try:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS "
                        f"SELECT * FROM read_parquet('{patron}')")
        except Exception:
            print(f"  [!] no disponible: {t}")
    return con


def componer(con):
    """Dos paneles por figura, como en la version revisada del documento.

    Una figura por gráfico obligaba a LaTeX a flotarlas de una en una y cada
    una ocupaba casi una plana; emparejadas caben en el flujo del texto y se
    leen comparativamente, que es como se comentan en el capítulo.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.2))
    panel_embudo(con, a1)
    a1.set_title("Efecto de las reglas de alcance")
    panel_nodos(con, a2)
    a2.set_title("Nodos por tipo y cobertura del RUES")
    fig.tight_layout()
    guardar(fig, "fig_alcance_nodos.pdf")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.2))
    panel_vinculos(con, a1)
    a1.set_title("Aristas por tipo de vínculo")
    panel_competencia(con, a2)
    a2.set_title("Oferentes por proceso")
    fig.tight_layout()
    guardar(fig, "fig_vinculos_competencia.pdf")

    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    if panel_splink(con, ax) is not False:
        fig.tight_layout()
        guardar(fig, "fig_splink_pesos.pdf")
    else:
        plt.close(fig)


def main():
    if not os.path.isdir(rutas.parquet()):
        sys.exit("falta OUTPUTS/gold: ejecuta antes 02_gold.py y 05_nodos.py")
    con = abrir()
    print(f"Figuras -> {SALIDA}")
    componer(con)
    con.close()


if __name__ == "__main__":
    main()
