#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ejecuta el pipeline completo, o el tramo que se le pida.

Alternativa a recorrer los cuadernos a mano, útil en una máquina nueva o para
rehacer el análisis entero de una vez. Ejecuta los mismos módulos que ellos.

Cada etapa declara qué datos necesita, de modo que se avisa antes de empezar en
vez de fallar a mitad de camino.

Uso:
    python orquestador.py --listar
    python orquestador.py --desde grafo
    python orquestador.py --solo comparativa
    python orquestador.py --simular
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tfm import rutas                                        # noqa: E402

#: (clave, módulo, argumentos, qué necesita para poder correr, descripción)
#:
#: `necesita` es lo que se comprueba antes de arrancar:
#:   "bronze"  los JSONL crudos
#:   "gold"    los parquet limpios
#:   "modelos" los artefactos de resultados
ETAPAS = [
    ("bronze", "tfm.ingesta.bronze", [], "bronze",
     "Ingesta de los nueve conjuntos JSONL"),
    ("gold", "tfm.ingesta.gold", [], "bronze",
     "Modelo dimensional en parquet"),
    ("nodos", "tfm.identidad.nodos", [], "gold",
     "Resolución determinista de identidad"),
    ("entidades", "tfm.identidad.entidades", [], "gold",
     "Normalización de entidades contratantes"),
    ("splink", "tfm.identidad.splink_resolver", [], "gold",
     "Resolución probabilística Fellegi-Sunter"),
    ("seudonimo", "tfm.identidad.seudonimizar", [], "gold",
     "Seudonimización de los actores citados"),
    ("grafo", "tfm.grafo.features_sna", [], "gold",
     "Variables de red e índice de riesgo"),
    ("comparativa", "tfm.modelos.comparativa", [], "modelos",
     "Los cuatro modelos sobre la misma partición"),
    ("fuga", "tfm.modelos.fuga_variantes", [], "modelos",
     "Variantes V1 y V2 y matriz de arquitecturas"),
    ("clasico", "tfm.modelos.brazo_clasico", [], "modelos",
     "Ordenamientos clásicos sin entrenar"),
    ("gnn", "tfm.modelos.gnn_paso_mensajes", ["--fase", "tabla"], "modelos",
     "GCN y GraphSAGE con paso de mensajes"),
    ("solapamiento", "tfm.modelos.solapamiento", [], "modelos",
     "Solapamiento entre los dos paradigmas"),
    ("operativas", "tfm.modelos.metricas_operativas", [], "modelos",
     "Métricas de lista corta y calibración"),
    ("tablas", "tfm.modelos.tablas_seudonimizadas", [], "modelos",
     "Tablas publicables de casos concretos"),
    ("gcn_sgc", "tfm.modelos.gcn_vs_sgc", [], "modelos",
     "Contraste entre la GCN y la simplificación que la publica"),
    ("explicabilidad", "tfm.modelos.explicabilidad", [], "modelos",
     "Explicación de las cien alertas de mayor puntuación"),
    ("incertidumbre", "tfm.modelos.incertidumbre", [], "modelos",
     "Intervalos de confianza de las métricas, por remuestreo"),
    ("figuras", "tfm.figuras.generales", [], "modelos",
     "Figuras de distribuciones y métricas"),
    ("figuras_red", "tfm.figuras.red", [], "modelos",
     "Figuras de estructura de red"),
    ("figuras_tripartita", "tfm.figuras.ejemplo_tripartito", [], "gold",
     "Figura de la estructura tripartita entidad-UT-socio"),
    ("figuras_convergencia", "tfm.figuras.convergencia", [], "modelos",
     "Figura de convergencia de las dos redes con paso de mensajes"),
    ("figuras_explicabilidad", "tfm.figuras.explicabilidad", [], "modelos",
     "Figura de la explicación de las alertas"),
    ("sensibilidad", "tfm.verificacion.sensibilidad", ["--que", "ambas"],
     "modelos", "Sensibilidad del criterio de exclusión y atribución por decil"),
    ("triangulos", "tfm.grafo.triangulos", [], "gold",
     "Recuento de triángulos de riesgo entidad-UT-socio"),
    ("exportar_tablas", "tfm.tablas_markdown", [], "modelos",
     "Exportación de las tablas publicadas a docs/tablas/"),
]


def disponible(necesita):
    import glob
    if necesita == "bronze":
        return bool(glob.glob(os.path.join(rutas.bronze(), "*")))
    if necesita == "gold":
        return os.path.isdir(rutas.parquet())
    if necesita == "modelos":
        return bool(glob.glob(rutas.modelos("*.csv")))
    return True


def listar():
    print(rutas.describe(), "\n")
    print(f"{'etapa':14} {'necesita':9} {'':3} descripción")
    print("-" * 78)
    for clave, _, _, necesita, desc in ETAPAS:
        marca = "ok " if disponible(necesita) else "NO "
        print(f"{clave:14} {necesita:9} {marca} {desc}")
    print("\n«NO» significa que faltan los datos de entrada de esa etapa; "
          "revisa el README.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--desde", help="clave de etapa por la que empezar")
    ap.add_argument("--hasta", help="clave de etapa en la que parar")
    ap.add_argument("--solo", help="una sola etapa")
    ap.add_argument("--simular", action="store_true")
    a = ap.parse_args()

    if a.listar:
        listar()
        return 0

    claves = [e[0] for e in ETAPAS]
    if a.solo:
        seleccion = [e for e in ETAPAS if e[0] == a.solo]
    else:
        i = claves.index(a.desde) if a.desde else 0
        j = claves.index(a.hasta) + 1 if a.hasta else len(ETAPAS)
        seleccion = ETAPAS[i:j]
    if not seleccion:
        sys.exit("no hay ninguna etapa que case; usa --listar")

    #: Se comprueban todas las dependencias antes de arrancar ninguna etapa.
    faltan = [(c, n) for c, _, _, n, _ in seleccion if not disponible(n)]
    if faltan and not a.simular:
        print("No se puede empezar; faltan datos de entrada:\n")
        for c, n in faltan:
            print(f"  {c:14} necesita «{n}»")
        print(f"\nDatos esperados en: {rutas.entrada()}")
        print("Consulta el apartado sobre los datos del README.md.")
        return 1

    from tfm import ejecutar
    t0 = time.time()
    for clave, modulo, args, _, desc in seleccion:
        print(f"\n{'=' * 78}\n{clave.upper()}  ·  {desc}\n{'=' * 78}")
        if a.simular:
            print(f"  [simulado] {modulo} {' '.join(args)}")
            continue
        ejecutar.correr(modulo, *args)
    print(f"\n{len(seleccion)} etapas en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
