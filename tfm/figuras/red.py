#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generar_figuras_red.py — Figuras de red del Capítulo 5.

Este script produce las visualizaciones propiamente de grafo: los componentes
conexos pequeños de la red tripartita entidad-UT-socio, el entorno del
componente gigante, la ego-red de una entidad con triángulos de riesgo y la
distribución de grado.

Se lee directamente `OUTPUTS/gold`, no el fichero .duckdb, porque este último
almacena rutas absolutas de Windows que no resuelven en otros entornos.

Salida: OUTPUTS/figuras/*.pdf, en vectorial
"""

import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
SALIDA = rutas.figuras_publicadas()

AZUL = "#0097CC"
GRIS = "#58595B"
NARANJA = "#D95F02"
VERDE = "#1B9E77"

#: Paleta por tipo de nodo de la red tripartita.
COLOR = {"entidad": "#1F4E79", "ut": NARANJA, "socio": VERDE}

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
    "grid.alpha": 0.22,
    "figure.dpi": 200,
})


def guardar(fig, nombre):
    os.makedirs(SALIDA, exist_ok=True)
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {nombre}")


def limpiar(ax):
    """Un diagrama de red no lleva ejes ni rejilla."""
    ax.set_axis_off()


# --------------------------------------------------------------------------
# Construcción de la red tripartita entidad - unión temporal - socio
# --------------------------------------------------------------------------

#: Divisiones CIIU de la sección K (actividades financieras y de seguros). Los
#: aseguradores garantizan pólizas en cientos de procesos sin relación entre sí
#: y actúan como concentradores artificiales: se excluyen de la red tripartita.
CIIU_FINANCIERO = ("64", "65", "66")

#: Las uniones temporales y buena parte de los proveedores no cruzan con el
#: registro mercantil y por tanto no tienen CIIU. Para esos casos se recurre al
#: patron de la razon social, que en el sector es muy estable.
SEMILLA = 42
#: El criterio de exclusión de banca y seguros vive en `tfm.datos`, para que
#: las tres etapas que lo aplican no puedan divergir.
from tfm.datos import condicion_banca_seguros    # noqa: E402


def cargar_tripartita(con):
    criterio = condicion_banca_seguros(con, f"{PQ}/nodo_2025/*.parquet")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW excluidos AS
        SELECT id_nodo FROM '{PQ}/nodo_2025/*.parquet'
        WHERE {criterio};

        CREATE OR REPLACE TEMP VIEW adj AS
        SELECT origen AS entidad, destino AS proveedor
        FROM '{PQ}/vinculo_2025/*.parquet'
        WHERE tipo_vinculo = 'adjudica'
          AND destino NOT IN (SELECT id_nodo FROM excluidos);

        CREATE OR REPLACE TEMP VIEW inte AS
        SELECT origen AS ut, destino AS socio
        FROM '{PQ}/vinculo_2025/*.parquet'
        WHERE tipo_vinculo = 'integra'
          AND destino NOT IN (SELECT id_nodo FROM excluidos)
          AND origen  NOT IN (SELECT id_nodo FROM excluidos)
          -- La unión temporal y el socio han de ser actores distintos; la
          -- razón está en `tfm/grafo/triangulos.py`.
          AND origen <> destino;
    """)

    #: Solo entran las uniones temporales que además recibieron adjudicación:
    #: una UT sin contrato en el ejercicio no aporta estructura de riesgo.
    aristas_eu = con.execute("""
        SELECT DISTINCT a.entidad, a.proveedor AS ut
        FROM adj a JOIN (SELECT DISTINCT ut FROM inte) i ON a.proveedor = i.ut
        ORDER BY 1, 2
    """).fetchall()
    aristas_us = con.execute("""
        SELECT DISTINCT i.ut, i.socio
        FROM inte i JOIN (SELECT DISTINCT proveedor FROM adj) a ON i.ut = a.proveedor
        ORDER BY 1, 2
    """).fetchall()
    #: Adjudicación directa de la entidad a un socio: es el tercer lado del
    #: triángulo de riesgo.
    aristas_es = con.execute("""
        SELECT DISTINCT a.entidad, a.proveedor AS socio
        FROM adj a JOIN (SELECT DISTINCT socio FROM inte) i ON a.proveedor = i.socio
        ORDER BY 1, 2
    """).fetchall()

    #: Las tres consultas anteriores llevan ORDER BY y no es cosmético.
    #: DuckDB paraleliza y, sin orden explícito, `fetchall()` devuelve las
    #: aristas en un orden distinto en cada ejecución. Ese orden es el orden de
    #: inserción de los nodos en el grafo, y Louvain **depende del orden de
    #: recorrido**: la semilla fija la aleatoriedad del algoritmo, no el grafo
    #: que recibe. El síntoma fue una modularidad que cambiaba entre
    #: ejecuciones —0,4825 frente a 0,5043 sobre la misma red— y descuadraba
    #: una tabla ya publicada. Ordenar aquí es lo que hace determinista todo lo
    #: que viene después.

    #: La clase de cada nodo se fija ANTES de montar el grafo. Si se asignase
    #: al vuelo, un actor que es socio en una unión y adjudicatario en otra
    #: acabaría con la etiqueta del último enlace insertado, y las figuras
    #: mostrarían un socio donde hay una unión temporal.
    clase = {}
    for e, u in aristas_eu:
        clase[e] = "entidad"
    for u, _ in aristas_us:
        clase[u] = "ut"
    for _, u in aristas_eu:
        clase.setdefault(u, "ut")
    for _, s in aristas_us:
        clase.setdefault(s, "socio")

    G = nx.Graph()
    for a, b in list(aristas_eu) + list(aristas_us) + list(aristas_es):
        #: Un lazo sobre el propio nodo no representa ninguna relación real:
        #: procede de registros en los que la unión temporal y su socio
        #: comparten documento.
        if a == b:
            continue
        if a not in clase or b not in clase:
            continue
        G.add_node(a, clase=clase[a])
        G.add_node(b, clase=clase[b])
        G.add_edge(a, b)

    #: El grafo se reconstruye con los nodos y las aristas en orden alfabético.
    #: Parece redundante —las consultas ya llevan ORDER BY— y no lo es: entre
    #: la consulta y aquí intervienen conjuntos de Python, cuyo orden de
    #: recorrido depende de la aleatorización de `hash()` y **cambia entre
    #: procesos**, no dentro de uno. Louvain recorre los nodos en el orden en
    #: que están almacenados, de modo que ese detalle bastaba para que la
    #: modularidad de una misma red saliera 0,4825 en una ejecución y 0,5043 en
    #: la siguiente. Con el grafo canonizado, dos ejecuciones dan lo mismo.
    H = nx.Graph()
    for n in sorted(G.nodes()):
        H.add_node(n, **G.nodes[n])
    for a, b in sorted(tuple(sorted(e)) for e in G.edges()):
        H.add_edge(a, b)
    return H


def dibujar(G, pos, ax, tam=14, ancho=0.35, alpha_e=0.35):
    nx.draw_networkx_edges(G, pos, ax=ax, width=ancho, edge_color="#9AA0A6",
                           alpha=alpha_e)
    for clase in ("socio", "ut", "entidad"):
        nodos = [n for n, d in G.nodes(data=True) if d.get("clase") == clase]
        if not nodos:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=nodos, ax=ax,
                               node_size=tam, node_color=COLOR[clase],
                               linewidths=0)
    limpiar(ax)


def leyenda(fig, ax=None):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4,
                      color=COLOR[c], label=e)
               for c, e in (("entidad", "Entidad"), ("ut", "Unión temporal"),
                            ("socio", "Socio"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def fig_componentes(G):
    """Componentes conexos pequeños: micro-mercados cerrados."""
    comps = sorted(nx.connected_components(G), key=len)
    chicos = [c for c in comps if 4 <= len(c) <= 9][:3]
    if len(chicos) < 3:
        chicos = [c for c in comps if len(c) >= 3][:3]

    fig, axes = plt.subplots(1, 3, figsize=(5.8, 1.9))
    for ax, comp in zip(axes, chicos):
        H = G.subgraph(comp)
        pos = nx.kamada_kawai_layout(H)
        dibujar(H, pos, ax, tam=42, ancho=0.7, alpha_e=0.65)
        ax.set_title(f"{H.number_of_nodes()} nodos, {H.number_of_edges()} enlaces")
    leyenda(fig)
    fig.tight_layout()
    guardar(fig, "fig_red_componentes.pdf")
    return len(comps)


def fig_gigante(G):
    """Entorno de las entidades de mayor grado dentro del componente gigante."""
    gigante = max(nx.connected_components(G), key=len)
    H = G.subgraph(gigante)
    ents = sorted((n for n, d in H.nodes(data=True) if d["clase"] == "entidad"),
                  key=lambda n: H.degree(n), reverse=True)[:8]
    vecinos = set(ents)
    for e in ents:
        vecinos |= set(H.neighbors(e))
    #: Un salto más, para que se vean los socios que cuelgan de cada UT.
    for u in list(vecinos):
        if H.nodes[u]["clase"] == "ut":
            vecinos |= set(H.neighbors(u))
    S = H.subgraph(vecinos)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    pos = nx.spring_layout(S, seed=42, k=0.32, iterations=60)
    dibujar(S, pos, ax, tam=9, ancho=0.22, alpha_e=0.3)
    nx.draw_networkx_nodes(S, pos, nodelist=ents, ax=ax, node_size=55,
                           node_color=COLOR["entidad"], edgecolors="white",
                           linewidths=0.5)
    leyenda(fig)
    fig.tight_layout()
    guardar(fig, "fig_red_gigante.pdf")
    return H.number_of_nodes(), H.number_of_edges(), len(gigante) / G.number_of_nodes()


def triangulos_sql(con):
    """Triangulos entidad-UT-socio segun la definicion de v_triangulo_2025.

    Se calcula en SQL y no sobre el objeto de grafo para que la cifra que
    aparece en el texto y la que ordena las figuras procedan de la misma
    definicion que emplea el pipeline.
    """
    con.execute("""
        CREATE OR REPLACE TEMP VIEW tri AS
        SELECT a.entidad, a.proveedor AS nodo_ut, i.socio AS nodo_socio
        FROM adj a
        JOIN inte i ON i.ut = a.proveedor
        JOIN adj  d ON d.entidad = a.entidad AND d.proveedor = i.socio;
    """)
    total, ents, uts, socios = con.execute("""
        SELECT count(*), count(DISTINCT entidad), count(DISTINCT nodo_ut),
               count(DISTINCT nodo_socio) FROM tri
    """).fetchone()
    ranking = con.execute("""
        SELECT entidad, count(*) n FROM tri GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    return total, ents, uts, socios, ranking


def fig_grado_tripartita(G):
    """Distribución de grado de la red tripartita depurada.

    Es la figura que el documento revisado por la dirección titulaba
    «Distribución de grado — red tripartita (sin banca/seguros)». Convive con
    la del grafo bipartito de proveedores y no la sustituye: son dos
    poblaciones distintas —15.411 nodos de tres tipos frente a 82.984
    proveedores— y su comparación es justamente lo informativo.
    """
    g = np.array([d for _, d in G.degree()])
    g = g[g > 0]
    vals, cts = np.unique(g, return_counts=True)
    clase = nx.get_node_attributes(G, "clase")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    a1.scatter(vals, cts, s=5, color=AZUL, alpha=0.7, edgecolors="none")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Grado del nodo"); a1.set_ylabel("N.º de nodos")
    a1.set_title("Distribución de grado (log-log)")

    #: El grado medio por tipo de nodo es lo que distingue a esta red de la
    #: bipartita: aquí conviven tres poblaciones con papeles muy distintos.
    tipos = [("entidad", "Entidades"), ("ut", "Uniones\ntemporales"),
             ("socio", "Socios")]
    medias, etiquetas = [], []
    for t, etq in tipos:
        gs = [d for n, d in G.degree() if clase.get(n) == t]
        if gs:
            medias.append(float(np.mean(gs))); etiquetas.append(etq)
    barras = a2.bar(range(len(medias)), medias,
                    color=[COLOR["entidad"], COLOR["ut"], COLOR["socio"]][:len(medias)])
    for b, v in zip(barras, medias):
        a2.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.2f}".replace(".", ","),
                ha="center", va="bottom", fontsize=6.5, color=GRIS)
    a2.set_xticks(range(len(etiquetas)))
    a2.set_xticklabels(etiquetas, fontsize=6.5)
    a2.set_ylabel("Grado medio")
    a2.set_title("Grado medio por tipo de nodo")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    guardar(fig, "fig_grado_tripartita.pdf")
    return {e: round(m, 2) for e, m in zip([t[0] for t in tipos], medias)}


def descriptores_ego(ego, centro, k, n_triangulos):
    """Métricas estructurales de una ego-red, en el mismo repertorio que la
    tabla de la red tripartita completa.

    Se calculan las mismas magnitudes que para la red entera —y no un
    subconjunto— porque el valor de estas dos tablas está en la comparación:
    una ego-red densa y una dispersa solo se distinguen si se miden con la
    misma vara. Las aristas no llevan etiqueta de tipo en el grafo, así que su
    naturaleza se deduce de la clase de sus dos extremos, que es exactamente
    como se construyeron.
    """
    clase = nx.get_node_attributes(ego, "clase")
    n_ent = sum(1 for v in clase.values() if v == "entidad")
    n_ut = sum(1 for v in clase.values() if v == "ut")
    n_soc = sum(1 for v in clase.values() if v == "socio")

    eu = us = es = 0
    for a, b in ego.edges():
        par = {clase.get(a), clase.get(b)}
        if par == {"entidad", "ut"}:
            eu += 1
        elif par == {"ut", "socio"}:
            us += 1
        elif par == {"entidad", "socio"}:
            es += 1

    grados = [d for _, d in ego.degree()]
    #: La asortatividad no está definida si todos los nodos tienen el mismo
    #: grado; en ese caso el denominador se anula y NetworkX devuelve NaN.
    try:
        asort = float(nx.degree_assortativity_coefficient(ego))
        if asort != asort:
            asort = None
    except Exception:
        asort = None
    nucleos = nx.core_number(ego)
    #: La ego-red se canoniza justo antes de detectar comunidades. Louvain
    #: recorre los nodos en el orden en que el grafo los almacena, y ese orden
    #: lo hereda de cómo se extrajo el subgrafo, no de la semilla: la semilla
    #: gobierna la aleatoriedad del algoritmo, no la del grafo que recibe.
    #: Sin este paso, la modularidad de una misma ego-red salía 0,4048 en una
    #: ejecución y 0,4008 en la siguiente, lo que descuadraba una tabla ya
    #: publicada del apartado 5.3.
    canon = nx.Graph()
    for n in sorted(ego.nodes()):
        canon.add_node(n, **ego.nodes[n])
    for a, b in sorted(tuple(sorted(e)) for e in ego.edges()):
        canon.add_edge(a, b)
    ego = canon

    comunidades = list(nx.community.louvain_communities(ego, seed=SEMILLA))

    return {
        "ego": f"Entidad {k}",
        "nodos": ego.number_of_nodes(),
        "entidades": n_ent,
        "uniones_temporales": n_ut,
        "socios": n_soc,
        "enlaces": ego.number_of_edges(),
        "enlaces_entidad_ut": eu,
        "enlaces_ut_socio": us,
        "enlaces_socio_entidad": es,
        "grado_medio": round(float(np.mean(grados)), 2),
        "grado_maximo": int(max(grados)),
        "densidad": round(float(nx.density(ego)), 4),
        "componentes_conexos": nx.number_connected_components(ego),
        "transitividad": round(float(nx.transitivity(ego)), 4),
        "clustering_medio": round(float(nx.average_clustering(ego)), 4),
        "asortatividad_grado": None if asort is None else round(asort, 4),
        "k_core_maximo": int(max(nucleos.values())),
        "triangulos_riesgo": int(n_triangulos),
        "comunidades_louvain": len(comunidades),
        "modularidad": round(float(nx.community.modularity(ego, comunidades)), 4),
        "socios_de_grado_1": sum(1 for n, d in ego.degree()
                                 if d == 1 and clase.get(n) == "socio"),
    }


def fig_triangulos(G, ranking):
    """Una figura por ego-red, con sus descriptores persistidos.

    Antes las dos ego-redes compartían figura y un solo comentario, de modo que
    el texto hablaba de «cada panel» sin decir nada de ninguno en particular.
    Separarlas permite analizarlas una a una, que es lo que pide un hallazgo
    que se presenta como accionable: si la promesa es que una alerta se explica
    en una frase, hay que escribir esa frase para cada caso.
    """
    cuenta = dict(ranking)
    elegidas = []
    for e, _ in ranking:
        if e not in G:
            continue
        ego = nx.ego_graph(G, e, radius=2)
        if 12 <= ego.number_of_nodes() <= 45:
            elegidas.append((e, ego))
        if len(elegidas) == 2:
            break

    filas = []
    for k, (e, ego) in enumerate(elegidas, 1):
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        pos = nx.kamada_kawai_layout(ego)
        dibujar(ego, pos, ax, tam=34, ancho=0.6, alpha_e=0.55)
        nx.draw_networkx_nodes(ego, pos, nodelist=[e], ax=ax, node_size=95,
                               node_color=COLOR["entidad"], edgecolors="white",
                               linewidths=0.7)
        ax.set_title(f"{cuenta[e]} triángulos de riesgo")
        leyenda(fig)
        fig.tight_layout()
        guardar(fig, f"fig_red_triangulos_{k}.pdf")

        filas.append(descriptores_ego(ego, e, k, cuenta[e]))

    destino = rutas.referencia("ego_redes_triangulos.csv")
    import csv as _csv
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"  fig_red_triangulos_1.pdf y _2.pdf  ->  {destino}")
    for f in filas:
        print(f"    {f['ego']}: {f['nodos']} nodos, {f['enlaces']} enlaces, "
              f"{f['triangulos_riesgo']} triángulos, densidad {f['densidad']}, "
              f"clustering {f['clustering_medio']}, k-core {f['k_core_maximo']}")
    return filas


def fig_grado_comunidades(con):
    """Distribución de grado en log-log y tamaño de las comunidades."""
    ruta = rutas.artefacto("features_nodo.parquet")
    g = con.execute(f"SELECT grado FROM '{ruta}'").fetchnumpy()["grado"]
    g = g[g > 0]
    vals, cts = np.unique(g, return_counts=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.1))
    a1.scatter(vals, cts, s=4, color=AZUL, alpha=0.65, edgecolors="none")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("Grado del nodo")
    a1.set_ylabel("N.º de nodos")
    a1.set_title("Distribución de grado (log-log)")

    #: El tamaño de comunidad se recalcula aquí solo si está disponible; en
    #: caso contrario se representa el reparto acumulado del grado, que
    #: transmite la misma idea de concentración.
    orden = np.sort(g)[::-1]
    acum = np.cumsum(orden) / orden.sum()
    x = np.arange(1, len(orden) + 1) / len(orden)
    a2.plot(x * 100, acum * 100, color=NARANJA, linewidth=1.2)
    a2.plot([0, 100], [0, 100], color=GRIS, linewidth=0.7, linestyle="--")
    a2.set_xlabel("% de nodos, del más conectado al menos")
    a2.set_ylabel("% de enlaces acumulados")
    a2.set_title("Concentración de los enlaces")
    fig.tight_layout()
    guardar(fig, "fig_grado_concentracion.pdf")

    p10 = acum[int(len(acum) * 0.10)] * 100
    return float(vals.max()), float(g.mean()), p10


def fig_curvas():
    """Curvas ROC, precision-exhaustividad e importancia de variables."""
    import pandas as pd
    #: Con `artefacto` y no contra `salidas/modelos/`: la figura tiene que
    #: poder dibujarse sobre un clon en el que solo esta el paquete de datos.
    roc = pd.read_csv(rutas.artefacto("curvas_roc.csv"))
    pr = pd.read_csv(rutas.artefacto("curvas_pr.csv"))
    #: Las etiquetas del CSV llevan el prefijo de familia; se conservan tal cual
    #: para que la figura no pueda desincronizarse de la salida del pipeline.
    etiqueta = ["SNA - Regresión Logística", "ML - Random Forest",
                "ML - Gradient Boosting", "SGC - propagación + MLP"]
    colores = dict(zip(etiqueta, [GRIS, AZUL, VERDE, NARANJA]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    for m in etiqueta:
        d = roc[roc["modelo"] == m]
        a1.plot(d["fpr"], d["tpr"], color=colores[m], linewidth=1.1, label=m)
        d = pr[pr["modelo"] == m]
        a2.plot(d["recall"], d["precision"], color=colores[m], linewidth=1.1)
    a1.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.7, linestyle="--")
    a1.set_xlabel("Tasa de falsos positivos")
    a1.set_ylabel("Tasa de verdaderos positivos")
    a1.set_title("Curva ROC")
    #: Sin la línea de prevalencia no se puede juzgar un PR-AUC: es el valor
    #: que obtendría un clasificador que ordenase al azar.
    #: La prevalencia es la precisión en el punto de exhaustividad máxima:
    #: cuando se señala a todo el mundo, se acierta en la proporción de
    #: positivos que hay. Es también el PR-AUC de un ordenamiento al azar.
    #: La prevalencia se lee de la curva ya cargada; la versión anterior
    #: releía el fichero contra una variable que no existe en este ámbito, de
    #: modo que la figura no llegaba a dibujarse.
    _pr = pr[pr["modelo"] == etiqueta[1]]
    prev = float(_pr.loc[_pr["recall"].idxmax(), "precision"])
    a2.axhline(prev, color="#BBBBBB", linewidth=0.8, linestyle="--")
    a2.text(0.98, prev + 0.03, f"prevalencia {prev:.4f}".replace(".", ","),
            ha="right", fontsize=5.8, color=GRIS)
    a2.set_xlabel("Exhaustividad")
    a2.set_ylabel("Precisión")
    a2.set_title("Curva precisión-exhaustividad")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=colores[m], linewidth=1.4, label=m)
                        for m in etiqueta],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    guardar(fig, "fig_curvas_modelos.pdf")

    #: Nombres legibles para el lector no tecnico; el fichero conserva el
    #: identificador interno de cada variable.
    NOMBRE = {
        "pagerank": "PageRank", "grado": "Grado",
        "strength": "Grado ponderado",
        "en_componente_gigante": "En componente gigante",
        "log_valor_total": "Valor contratado (log)",
        "log_valor_medio": "Valor medio por contrato (log)",
        "n_procesos": "N.º de procesos", "n_contratos": "N.º de contratos",
        "log_n_contratos": "N.º de contratos (log)",
        "n_entidades": "N.º de entidades",
        "n_cuentas_plataforma": "N.º de cuentas de plataforma",
    }
    #: La importancia se lee de la variante V1, la de las siete variables que
    #: sobreviven al control de fuga. La de las once incluye las cuatro desde
    #: las que la etiqueta se reconstruye, de modo que su lectura describe la
    #: fuga y no el modelo: interpretarla contradiría el apartado que la retira.
    imp = pd.read_csv(rutas.artefacto(os.path.join("v1", "importancia_rf.csv")))
    imp = imp.sort_values("importancia")
    imp["etiqueta"] = imp["feature"].map(lambda f: NOMBRE.get(f, f))
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    colores_b = [NARANJA if f in ("pagerank", "grado", "strength",
                                  "en_componente_gigante") else AZUL
                 for f in imp["feature"]]
    ax.barh(imp["etiqueta"], imp["importancia"] * 100, color=colores_b,
            height=0.68)
    ax.set_xlabel("Importancia (%)")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(imp["importancia"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NARANJA, label="Estructural (red)"),
                       Patch(color=AZUL, label="Volumen y actividad")],
              loc="lower right", frameon=False)
    fig.tight_layout()
    guardar(fig, "fig_importancia_rf.pdf")


def fig_lift():
    """Precisión y lift según el tamaño de la lista corta."""
    import pandas as pd
    d = pd.read_csv(rutas.artefacto("curva_lift.csv"))
    orden = ["SGC — propagación + MLP", "ML — Gradient Boosting",
             "ML — Random Forest", "SNA — Regresión Logística",
             "Trivial — Número de contratos", "Clásico — PageRank puro",
             "Clásico — Banderas rojas"]
    colores = [NARANJA, VERDE, AZUL, GRIS, "#8C6D31", "#7B68A6", "#C0392B"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    for nombre, color in zip(orden, colores):
        z = d[d["modelo"] == nombre].sort_values("k")
        if z.empty:
            continue
        a1.plot(z["k"], z["precision"], marker="o", ms=2.5, lw=1.1,
                color=color, label=nombre)
        a2.plot(z["k"], z["lift"], marker="o", ms=2.5, lw=1.1, color=color)
    prev = float(d["precision"].iloc[0] / d["lift"].iloc[0])
    a1.axhline(prev, color="#BBBBBB", lw=0.8, ls="--")
    a1.text(1000, prev + 0.02, "prevalencia", ha="right", fontsize=5.8, color=GRIS)
    a2.axhline(1, color="#BBBBBB", lw=0.8, ls="--")
    for a, t in ((a1, "Precisión@k"), (a2, "Lift@k")):
        a.set_xscale("log")
        a.set_xticks([50, 100, 200, 500, 1000])
        a.set_xticklabels(["50", "100", "200", "500", "1000"])
        a.set_xlabel("tamaño de la lista corta (k)")
        a.set_title(t)
    a1.set_ylim(0, 1.05)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=5.6,
               bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    guardar(fig, "fig_lift.pdf")


def fig_calibracion():
    """Diagrama de fiabilidad del mejor modelo."""
    import pandas as pd
    d = pd.read_csv(rutas.artefacto("tabla_5_28_calibracion.csv"))
    f = d.dropna(subset=["probabilidad media"])
    f = f[f["probabilidad media"] != ""]
    x = f["probabilidad media"].astype(float).values
    y = f["frecuencia observada"].astype(float).values
    n = f["n"].astype(float).values

    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8, ls="--")
    ax.plot(x, y, marker="o", ms=3, lw=1.2, color=NARANJA)
    #: El tamaño del punto informa de cuántos casos sostienen cada tramo: los
    #: extremos superiores tienen muy pocos y su desviación no es significativa.
    ax.scatter(x, y, s=8 + 40 * n / n.max(), color=NARANJA, zorder=3, alpha=0.6)
    ax.set_xlabel("Probabilidad predicha (media del tramo)")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Diagrama de fiabilidad")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    guardar(fig, "fig_calibracion.pdf")


def fig_solapamiento():
    """Las dos salidas señalan a poblaciones casi disjuntas."""
    import pandas as pd
    sol = pd.read_csv(rutas.artefacto("tabla_5_23_solapamiento.csv"))
    zon = pd.read_csv(rutas.artefacto("tabla_5_24_zonas.csv"))
    f = sol[sol["Configuración"] == "V1"].iloc[0]
    solo_m = int(f["solo modelo"]); ambos = int(f["ambos"])
    solo_t = int(f["solo triángulo"]); esperado = float(f["esperados si independientes"])

    #: Más alta y con el panel izquierdo algo más estrecho que el derecho. La
    #: versión anterior media 2,9 pulgadas de alto para dos paneles con texto
    #: dentro, y el resultado era ilegible: las celdas de la tabla cruzada
    #: quedaban aplastadas, la leyenda se montaba encima de las barras y los
    #: rótulos del eje se pisaban entre sí.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 4.0),
                                 gridspec_kw={"width_ratios": [1.18, 1.0]})

    #: Se descartaron un diagrama de conjuntos proporcional y una barra apilada:
    #: a media anchura de página la escala isométrica aplasta el panel y la
    #: barra deja la leyenda encima de los rótulos. Una tabla cruzada dibujada
    #: es lo que mejor se lee y es además lo que sostiene el contraste.
    from matplotlib.patches import Rectangle
    ninguno = int(f["ninguno"])
    obs = [[ambos, solo_m], [solo_t, ninguno]]
    esp = [[esperado, solo_m + ambos - esperado],
           [solo_t + ambos - esperado, 0]]
    for i in range(2):
        for j in range(2):
            destaca = (i == 0 and j == 0)
            a1.add_patch(Rectangle((j, -i), 1, 1, facecolor=(NARANJA if destaca
                                                             else "#F2F2F2"),
                                   alpha=0.75 if destaca else 1.0,
                                   edgecolor="white", linewidth=2))
            #: El cuerpo se ajusta al ancho de celda: a 12 puntos «23.685» y
            #: «1.011» se tocaban entre celdas contiguas.
            a1.text(j + 0.5, -i + 0.58, f"{obs[i][j]:,}".replace(",", "."),
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if destaca else GRIS)
            if i + j < 2:
                #: El rótulo va debajo del número y con la coma decimal del
                #: documento. A tamaño 7 los dos de la fila superior se
                #: solapaban entre celdas: «esperados 191,9» es más ancho que
                #: la celda que lo contiene.
                #: Abreviado: «esperados 191,9» a cuerpo legible es más ancho
                #: que la celda, y los rótulos de las dos celdas superiores
                #: se montaban uno sobre otro.
                a1.text(j + 0.5, -i + 0.28,
                        f"esp. {esp[i][j]:.1f}".replace(".", ","),
                        ha="center", va="center", fontsize=6.5,
                        color="white" if destaca else GRIS)
    a1.text(0.5, 1.10, "en triángulo", ha="center", fontsize=8)
    a1.text(1.5, 1.10, "no", ha="center", fontsize=8)
    a1.text(-0.06, 0.5, "en el top-200", ha="right", va="center", fontsize=8)
    a1.text(-0.06, -0.5, "no", ha="right", va="center", fontsize=8)
    #: Las tres magnitudes del contraste, en dos líneas para que quepan sin
    #: reducir el cuerpo: en una sola se salían del panel.
    a1.text(1.0, -1.30, f"Jaccard {f['Jaccard']:.4f}".replace(".", ",")
            + f"   ·   odds ratio {f['odds ratio']:.3f}".replace(".", ","),
            ha="center", fontsize=7.5)
    a1.text(1.0, -1.52, "contraste exacto de Fisher: p = "
            + f"{float(f['p (Fisher)']):.5f}".replace(".", ","),
            ha="center", fontsize=7.5)
    a1.set_xlim(-1.00, 2.06)
    a1.set_ylim(-1.75, 1.30)
    #: Sin escala isométrica: una tabla cruzada no necesita celdas cuadradas,
    #: y forzarla hace que tight_layout aplaste el panel a media anchura.
    limpiar(a1)
    a1.set_title("Tabla cruzada sobre el conjunto de prueba", pad=8)

    #: Panel derecho: por qué son disjuntos. Cuatro descriptores, dos zonas.
    zz = zon.set_index("Zona")
    campos = [("% uniones temporales", "% uniones\ntemporales", 1),
              ("Valor total (mediana, mill.)", "Valor mediano\n(mill. COP)", 1),
              ("% contratación directa (mediana)", "Contratación\ndirecta (×100)", 100),
              ("CRI (mediana)", "CRI mediano\n(×100)", 100)]
    etiquetas = [c[1] for c in campos]
    v_mod = [float(zz.loc["Solo el modelo", c]) * k for c, _, k in campos]
    v_tri = [float(zz.loc["Solo la señal estructural", c]) * k for c, _, k in campos]
    campos_llanos = [c[1].replace("\n", " ") for c in campos]
    y = np.arange(len(campos)); h = 0.38
    b1 = a2.barh(y + h / 2, v_mod, h, color=NARANJA, label="Solo el modelo")
    b2 = a2.barh(y - h / 2, v_tri, h, color=VERDE, label="Solo la estructura")
    a2.set_xscale("log")
    #: El límite se fija antes de rotular los ceros. Antes se escribían en 1,2
    #: sobre un eje cuyo mínimo elegía matplotlib por encima de ese valor, con
    #: lo que los dos «0» flotaban fuera del área de dibujo, desconectados de
    #: su barra y sin que se entendiera a qué se referían.
    tope = max(max(v_mod), max(v_tri))
    a2.set_xlim(0.6, tope * 2.4)
    for barras, vals in ((b1, v_mod), (b2, v_tri)):
        for b, v in zip(barras, vals):
            if v <= 0:
                a2.text(0.72, b.get_y() + b.get_height() / 2, "0",
                        ha="left", va="center", fontsize=7, color=GRIS,
                        fontweight="bold")
            else:
                a2.text(v * 1.12, b.get_y() + b.get_height() / 2,
                        f"{v:,.0f}".replace(",", "."), ha="left", va="center",
                        fontsize=6.5, color=GRIS)
    a2.set_yticks(y)
    a2.set_yticklabels(campos_llanos, fontsize=7.5)
    a2.invert_yaxis()
    a2.tick_params(axis="x", labelsize=7)
    a2.set_xlabel("Escala logarítmica", fontsize=7.5)
    #: La leyenda sale del área de dibujo: dentro tapaba las dos primeras
    #: barras, que son justamente las que sostienen la comparación.
    a2.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.3, columnspacing=1.2)
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    guardar(fig, "fig_solapamiento.pdf")


def main():
    con = duckdb.connect()
    print("Figuras de red del Capítulo 5")
    G = cargar_tripartita(con)
    print(f"  red tripartita: {G.number_of_nodes():,} nodos, "
          f"{G.number_of_edges():,} enlaces")
    n_comp = fig_componentes(G)
    n, m, frac = fig_gigante(G)
    tri, ents, uts, socios, ranking = triangulos_sql(con)
    egos = fig_triangulos(G, ranking)
    gmax, gmed, p10 = fig_grado_comunidades(con)
    gtri = fig_grado_tripartita(G)
    for nombre, f in (("lift", fig_lift), ("calibración", fig_calibracion)):
        try:
            f()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  ({nombre} omitida: {exc})")
    try:
        fig_solapamiento()
    except (FileNotFoundError, KeyError) as exc:
        print(f"  (solapamiento omitido: {exc})")
    try:
        fig_curvas()
    except FileNotFoundError:
        print("  (curvas omitidas: faltan las salidas de los modelos)")

    #: Estas ocho magnitudes las cita el apartado 5.3 y hasta ahora solo
    #: existían en la salida por consola de este script. Una cifra que no está
    #: en ningún fichero no es verificable, así que se persisten.
    import csv as _csv
    destino = rutas.referencia("topologia_tripartita.csv")
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["magnitud", "valor"])
        for k, v in (("Nodos de la red tripartita", G.number_of_nodes()),
                     ("Enlaces de la red tripartita", G.number_of_edges()),
                     ("Componentes conexos", n_comp),
                     ("Nodos de la componente gigante", n),
                     ("Enlaces de la componente gigante", m),
                     ("Fracción de la red en la componente gigante",
                      round(frac * 100, 2)),
                     ("Grado máximo", int(gmax)),
                     ("Grado medio", round(gmed, 2)),
                     ("Porcentaje de enlaces en el decil superior de grado",
                      round(p10, 1)),
                     ("Triángulos de riesgo", tri),
                     ("Entidades con triángulo", ents),
                     ("Uniones temporales con triángulo", uts),
                     ("Socios con triángulo", socios)):
            w.writerow([k, v])
    print(f"  -> {destino}")

    print(f"  grado medio tripartita por tipo: {gtri}")

    print("\nCifras para el texto")
    print(f"  componentes conexos        : {n_comp:,}")
    print(f"  componente gigante         : {n:,} nodos, {m:,} enlaces "
          f"({frac:.2%} de la red)")
    print(f"  triangulos de riesgo       : {tri:,} sobre {ents:,} entidades,\n                               {uts:,} UT y {socios:,} socios")
    print(f"  grado máximo / medio       : {gmax:,.0f} / {gmed:,.2f}")
    print(f"  enlaces en el 10% superior : {p10:,.1f} %")


if __name__ == "__main__":
    main()
