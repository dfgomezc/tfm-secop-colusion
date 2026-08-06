#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrasta lo que produce una ejecución con lo publicado en el estudio.

Una ejecución escribe en `salidas/modelos/` y nunca toca `INPUT/`, que es de
solo lectura. Eso permite recalcular un resultado y compararlo con el que trae
el paquete de datos, en vez de sobreescribirlo y perder la referencia.

    from tfm import comparar
    comparar.tabla("tabla_5_18_variantes.csv")   # una tabla
    comparar.todo()                              # lo recalculado en la sesión

La comparación de números usa una tolerancia, no igualdad exacta: las
bibliotecas cambian de versión y el último decimal de un AUC puede moverse sin
que el resultado sea otro. Lo que importa es si una cifra publicada sigue
sosteniéndose, no si coincide bit a bit.
"""

import os

from tfm import rutas

#: Diferencia relativa por debajo de la cual dos números se consideran el mismo
#: resultado. 1e-6 distingue un cambio real de un redondeo distinto, y no
#: señala como discrepancia lo que el documento publica con cuatro decimales.
TOLERANCIA = 1e-6


def _leer(ruta):
    import pandas as pd
    if ruta.endswith(".parquet"):
        return pd.read_parquet(ruta)
    return pd.read_csv(ruta)


def tabla(nombre, tolerancia=TOLERANCIA, mostrar=True):
    """Compara una tabla recalculada con la publicada.

    Devuelve un diccionario con el veredicto. Si la tabla no se ha recalculado
    en esta sesión, lo dice en vez de comparar un fichero consigo mismo.
    """

    propia = rutas.resultados(nombre)
    publicada = rutas.referencia(nombre)

    if not os.path.exists(propia):
        veredicto = {"tabla": nombre, "estado": "no recalculada"}
    elif not os.path.exists(publicada):
        veredicto = {"tabla": nombre, "estado": "nueva, no está en el paquete"}
    else:
        a, b = _leer(propia), _leer(publicada)
        #: Se alinean por su clave antes de comparar. Dos tablas con el mismo
        #: contenido pueden salir en distinto orden de filas —una consulta sin
        #: ORDER BY no garantiza ninguno—, y compararlas posición a posición
        #: señalaría como diferencia lo que solo es un orden distinto.
        for clave in ("id_nodo", "ego", "modelo", "ordenamiento", "variante"):
            if clave in a.columns and clave in b.columns:
                a = a.sort_values(clave).reset_index(drop=True)
                b = b.sort_values(clave).reset_index(drop=True)
                break
        if list(a.columns) != list(b.columns):
            veredicto = {"tabla": nombre, "estado": "columnas distintas",
                         "detalle": f"{list(a.columns)} vs {list(b.columns)}"}
        elif len(a) != len(b):
            veredicto = {"tabla": nombre, "estado": "filas distintas",
                         "detalle": f"{len(a)} vs {len(b)}"}
        else:
            num = a.select_dtypes("number").columns
            difs = []
            for c in num:
                #: Se compara en relativo sobre el valor publicado, con
                #: resguardo para el cero.
                d = (a[c] - b[c]).abs() / b[c].abs().clip(lower=1e-12)
                if (d > tolerancia).any():
                    peor = d.idxmax()
                    difs.append(f"{c}: fila {peor}, "
                                f"{a[c].iloc[peor]} vs {b[c].iloc[peor]}")
            texto = [c for c in a.columns if c not in num]
            for c in texto:
                if not a[c].astype(str).equals(b[c].astype(str)):
                    difs.append(f"{c}: difiere en texto")
            veredicto = {"tabla": nombre,
                         "estado": "coincide" if not difs else "difiere",
                         "detalle": "; ".join(difs[:4])}

    if mostrar:
        marca = {"coincide": "=", "difiere": "!", "no recalculada": "-"}.get(
            veredicto["estado"], "?")
        print(f"  {marca} {nombre:46} {veredicto['estado']}"
              + (f"  ({veredicto.get('detalle','')[:70]})"
                 if veredicto.get("detalle") else ""))
    return veredicto


def todo(tolerancia=TOLERANCIA):
    """Compara todo lo que esta sesión haya recalculado.

    Solo mira los ficheros que existen en `salidas/modelos/`: lo que no se ha
    vuelto a calcular no se compara, porque compararlo sería contrastar el
    paquete de datos consigo mismo.
    """
    import glob

    base = rutas.resultados()
    nombres = sorted(os.path.relpath(r, base)
                     for r in glob.glob(os.path.join(base, "**", "*.csv"),
                                        recursive=True))
    nombres += sorted(os.path.relpath(r, base)
                      for r in glob.glob(os.path.join(base, "*.parquet")))
    if not nombres:
        print("No hay nada recalculado en salidas/modelos/.\n"
              "Ejecuta antes la celda que reconstruye el resultado.")
        return []

    print(f"{len(nombres)} artefactos recalculados, contra el paquete de datos:\n")
    veredictos = [tabla(n, tolerancia) for n in nombres]
    cuenta = {}
    for v in veredictos:
        cuenta[v["estado"]] = cuenta.get(v["estado"], 0) + 1
    print("\n" + " · ".join(f"{v} {k}" for k, v in sorted(cuenta.items())))
    return veredictos


def publicar(nombre=None):
    """Copia lo recalculado sobre el paquete de datos. **Destructivo.**

    Sustituye el artefacto publicado por el de esta ejecución, de modo que la
    referencia contra la que se compara deja de ser la del estudio. En los
    cuadernos esta llamada va comentada a propósito: se usa solo cuando se
    quiere publicar un resultado nuevo, y conviene hacerlo con el repositorio
    limpio para que el cambio quede en el control de versiones.
    """
    import glob
    import shutil

    base = rutas.resultados()
    if nombre:
        origenes = [rutas.resultados(nombre)]
    else:
        origenes = sorted(glob.glob(os.path.join(base, "**", "*"),
                                    recursive=True))
        origenes = [o for o in origenes if os.path.isfile(o)]

    copiados = 0
    for o in origenes:
        rel = os.path.relpath(o, base)
        d = rutas.referencia(rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(o, d)
        copiados += 1
        print(f"  publicado {rel}")
    print(f"\n{copiados} artefactos sustituidos en {rutas.referencia()}")
    return copiados
