#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprueba que la máquina puede ejecutar el pipeline, y qué parte de él.

Se lanza antes que nada, en un repositorio recién descomprimido:

    python comprobar_entorno.py

No calcula nada ni escribe nada. Responde tres preguntas en orden:

  1. ¿Están el intérprete y las dependencias?
  2. ¿Está el árbol de carpetas donde el código lo espera, y qué datos hay?
  3. ¿Qué etapas del orquestador pueden correr **hoy**, con lo que hay?

La tercera es la que importa, y por eso este guion existe además de
`orquestador.py --listar`: aquel resuelve la disponibilidad de la ingesta
mirando si ya existe `OUTPUTS/bronze/`, de modo que en una máquina limpia marca
«NO» justo la etapa que hay que lanzar primero. Aquí la ingesta se juzga por lo
que de verdad necesita, que es `INPUT/`.
"""

import importlib
import os
import shutil
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

OK, NO, AVISO = "  [ok]  ", "  [NO]  ", "  [ ! ]  "

#: Lo que pide `requirements.txt`, por el nombre con que se importa.
#: (módulo, imprescindible, para qué). Lo que no es imprescindible bloquea una
#: etapa concreta, no el pipeline: se avisa, no se detiene.
PAQUETES = [("duckdb", True, "motor analítico"),
            ("pandas", True, ""),
            ("numpy", True, ""),
            ("pyarrow", True, "lectura de parquet"),
            ("sklearn", True, "scikit-learn"),
            ("scipy", True, "matrices dispersas del paso de mensajes"),
            ("networkx", True, "métricas de red"),
            ("matplotlib", True, "figuras"),
            ("shap", False, "solo para la explicabilidad del OE-6"),
            ("splink", False, "solo para la resolución probabilística"),
            ("nbconvert", False, "solo para ejecutar los cuadernos")]

#: Programas externos que no instala pip. Solo hacen falta para el PDF.
PROGRAMAS = []

#: Qué necesita de verdad cada etapa, frente a lo que declara el orquestador.
NECESITA_REAL = {"bronze": "input"}


def titulo(t):
    print(f"\n{t}\n" + "-" * len(t))


def paso_1():
    titulo("1. Intérprete y dependencias")
    v = sys.version_info
    marca = OK if (v.major, v.minor) >= (3, 10) else AVISO
    print(f"{marca}Python {v.major}.{v.minor}.{v.micro}"
          + ("" if marca == OK else "  (el pipeline se validó con 3.11)"))
    faltan, opcionales = [], []
    for mod, clave, para in PAQUETES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"{OK}{mod:12} {ver:10} {para}")
        except Exception:
            (faltan if clave else opcionales).append(mod)
            print(f"{NO if clave else AVISO}{mod:12} {'':10} {para}")
    for prog, para in PROGRAMAS:
        hay = shutil.which(prog)
        print(f"{OK if hay else AVISO}{prog:12} {'':10} {para}"
              + ("" if hay else "  (no está en el PATH)"))
    if faltan:
        print(f"\n{NO}faltan {len(faltan)} imprescindibles: "
              "pip install -r requirements.txt")
    if opcionales:
        print(f"{AVISO}sin {', '.join(opcionales)}: el resto del pipeline corre "
              "igual, pero esas etapas no.")
    return not faltan


def paso_2():
    titulo("2. Árbol de carpetas y datos disponibles")
    try:
        from tfm import rutas
    except Exception as e:
        print(f"{NO}no se puede importar `tfm`: {e}")
        print("      Lánzalo desde la raíz del repositorio.")
        return None
    print(f"{OK}raíz  : {rutas.raiz()}")
    print(f"{OK}INPUT : {rutas.entrada()}")
    print(f"{OK}OUTPUT: {rutas.outputs()}")

    import glob
    CONJUNTOS = ["procesos_contratacion_s2", "contratos_electronicos_s2",
                 "proponentes_por_proceso_s2", "ofertas_por_proceso_s2",
                 "grupos_proveedores_s2", "proveedores_registrados_s2",
                 "datos_de_contacto_s2", "multas_sanciones_s2",
                 "matriculas_rues"]
    print("\n  Conjuntos de origen bajo INPUT/:")
    hay_input = 0
    for c in CONJUNTOS:
        d = rutas.entrada_de(c)
        n = len(glob.glob(os.path.join(d, "*.jsonl")))
        z = len(glob.glob(os.path.join(d, "*.zip"))) + \
            len(glob.glob(os.path.join(rutas.entrada(), f"{c}*.zip")))
        if n:
            hay_input += 1
            print(f"{OK}{c:30} {n:4} .jsonl")
        elif z:
            print(f"{AVISO}{c:30} {z:4} .zip sin descomprimir"
                  "  ->  python desempaquetar_jsonl.py")
        else:
            print(f"{NO}{c:30}    —")

    print("\n  Temas de OUTPUTS/:")
    estado = {}
    for t in ("bronze", "gold", "graph_sna", "modelos", "tablas", "curvas",
              "figuras"):
        d = rutas.outputs_de(t)
        n = sum(len(f) for _, _, f in os.walk(d)) if os.path.isdir(d) else 0
        estado[t] = n
        print(f"{OK if n else NO}{t:12} {n:6} ficheros"
              + ("" if n else "   (vacío)"))
    return {"input": hay_input, **estado}


def paso_3(estado):
    titulo("3. Etapas que pueden correr ahora")
    if estado is None:
        return
    try:
        import orquestador
    except Exception as e:
        print(f"{NO}no se puede importar el orquestador: {e}")
        return

    def disponible(clave, necesita):
        necesita = NECESITA_REAL.get(clave, necesita)
        if necesita == "input":
            return estado["input"] > 0
        if necesita == "bronze":
            return estado["bronze"] > 0
        if necesita == "gold":
            return estado["gold"] > 0
        if necesita == "modelos":
            return estado["modelos"] > 0 or estado["graph_sna"] > 0
        return True

    listas, bloqueadas = [], []
    for clave, modulo, _, necesita, desc in orquestador.ETAPAS:
        (listas if disponible(clave, necesita) else bloqueadas).append(
            (clave, NECESITA_REAL.get(clave, necesita), desc))
    for clave, nec, desc in listas:
        print(f"{OK}{clave:22} {desc}")
    for clave, nec, desc in bloqueadas:
        print(f"{NO}{clave:22} {desc}   (le falta: {nec})")

    print()
    if not listas:
        print("  Nada puede correr todavía. Empieza por dejar los .jsonl en "
              "INPUT/,\n  o por descomprimir el paquete de datos sobre OUTPUTS/.")
    elif bloqueadas:
        print(f"  {len(listas)} de {len(listas) + len(bloqueadas)} etapas "
              "disponibles. La guía de ejecución explica\n"
              "  por dónde entrar según lo que tengas: README.md")
    else:
        print("  Todas las etapas disponibles: python orquestador.py --simular")


def main():
    print("=" * 70)
    print("  TFM SECOP II — comprobación del entorno")
    print("=" * 70)
    ok = paso_1()
    estado = paso_2() if ok else None
    paso_3(estado)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
