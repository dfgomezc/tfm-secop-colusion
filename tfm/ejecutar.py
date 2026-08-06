#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invocar los módulos del pipeline desde un cuaderno y mostrar sus salidas.

Los cuadernos no reimplementan nada: llaman al módulo que hace el trabajo y
enseñan lo que produce.

  `correr`     ejecuta un módulo como si se lanzara desde la línea de órdenes
  `tabla`      carga una tabla de resultados publicada
  `figura`     muestra una figura ya generada
  `situacion`  cabecera con las rutas de la ejecución
"""

import contextlib
import glob
import io
import os
import runpy
import sys
import time

from tfm import rutas


def correr(modulo, *args, silencioso=False):
    """Ejecuta `modulo` como `__main__` con los argumentos dados.

    Se usa `runpy` en vez de `subprocess` para que el módulo herede el
    intérprete y las variables de entorno del cuaderno, en particular
    `TFM_OUTPUTS`, que decide de qué volcado se lee.

    Devuelve el texto que el módulo imprimió, además de mostrarlo.
    """
    argv = [modulo.split(".")[-1], *[str(a) for a in args]]
    guardado, sys.argv = sys.argv, argv
    buffer = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buffer):
            try:
                runpy.run_module(modulo, run_name="__main__")
            except SystemExit as e:
                #: Los módulos terminan con `SystemExit`; sin capturarlo, un
                #: código distinto de cero cerraría el núcleo del cuaderno.
                if e.code:
                    print(f"\n[el módulo terminó con código {e.code}]")
    finally:
        sys.argv = guardado
    salida = buffer.getvalue()
    if not silencioso:
        print(salida)
    print(f"[{modulo} · {time.time() - t0:.1f} s]")
    return salida


def tabla(nombre, n=None):
    """Un CSV publicado en `OUTPUTS/`, como DataFrame.

    Acepta el nombre con o sin extensión y admite coincidencia parcial.
    """
    import pandas as pd
    patron = nombre if nombre.endswith(".csv") else f"*{nombre}*.csv"
    #: Se busca por los temas de `OUTPUTS/` y también en `salidas/modelos/`,
    #: y este último primero: si la sesión ha recalculado algo, es eso lo que
    #: interesa mirar. `rutas.artefacto` no sirve aquí porque resuelve un
    #: nombre concreto y esto admite coincidencia parcial.
    bases = [rutas.resultados()] + [rutas.outputs_de(t) for t in rutas.ORDEN_TEMAS]
    rutas_csv = []
    for base in bases:
        rutas_csv += sorted(glob.glob(os.path.join(base, patron)))
        rutas_csv += sorted(glob.glob(os.path.join(base, "*", patron)))
    if not rutas_csv:
        raise FileNotFoundError(
            f"no hay ningún CSV que case con «{patron}» en {rutas.outputs()}.\n"
            f"¿Está descomprimido el paquete de datos en {rutas.outputs()}?")
    df = pd.read_csv(rutas_csv[0])
    return df.head(n) if n else df


def figura(nombre):
    """Muestra una figura ya generada."""
    from IPython.display import Image, display
    ruta = rutas.figuras_publicadas(nombre)
    if ruta.endswith(".pdf"):
        #: Las figuras se generan en PDF vectorial; en el cuaderno se muestra
        #: el PNG equivalente si existe.
        alterna = ruta[:-4] + ".png"
        ruta = alterna if os.path.exists(alterna) else ruta
    if not os.path.exists(ruta):
        print(f"falta {ruta}; ejecuta el cuaderno de figuras")
        return
    display(Image(filename=ruta))


def situacion():
    """Cabecera que todo cuaderno imprime al arrancar."""
    print(rutas.describe())
    if not rutas.existe_entrada():
        print("\nFaltan los datos. Descomprime el paquete según el README, "
              "o apunta TFM_INPUT a donde ya estén.")
    return rutas.existe_entrada()
