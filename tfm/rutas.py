#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rutas del proyecto.

Todos los módulos resuelven aquí dónde leer y dónde escribir, en vez de calcular
la raíz contando directorios. La raíz se localiza subiendo hasta encontrar
`pyproject.toml`, de modo que da igual desde dónde se importe: un script, un
cuaderno o el orquestador.

Convenio de directorios:

    INPUT/          los JSONL de origen, tal como se descargan del portal
    OUTPUTS/        todo lo que produce el procesamiento, por temas
      bronze/         los nueve conjuntos de origen, ya en parquet
      gold/           el modelo dimensional
      graph_sna/      la red, sus variables y el índice de riesgo
      modelos/        particiones, predicciones, rejillas y puntos de control
      tablas/         las tablas publicadas
      curvas/         las series de las que salen las figuras
      figuras/        las figuras generadas
    docs/           tablas exportadas a markdown
    salidas/        lo que produce *esta* ejecución, para contrastar

La frontera está en `OUTPUTS/`: el procesamiento de los JSONL termina ahí y
todo el análisis parte de esos parquet, no de los JSONL. `INPUT` no hace falta
para reproducir los resultados; `OUTPUTS` sí.

`salidas/` es el área de recálculo: una ejecución escribe ahí y se compara con
`OUTPUTS/` en vez de sobreescribirlo.
"""

import os

#: Fichero que marca la raíz del repositorio. Se busca hacia arriba desde este
#: módulo. Es preferible a contar niveles porque sobrevive a que el paquete se
#: reorganice.
MARCADOR = "pyproject.toml"

#: Variables de entorno que permiten trabajar contra datos externos al árbol.
#: `TFM_INPUT` es la útil en la práctica: los JSONL de origen ocupan decenas de
#: gigabytes y suelen descomprimirse en otro disco.
VAR_RAIZ = "TFM_RAIZ"
VAR_INPUT = "TFM_INPUT"
VAR_OUTPUTS = "TFM_OUTPUTS"

#: Qué tema guarda cada artefacto. El reparto es por función, no por formato:
#: `features_nodo.parquet` es una salida de la etapa de red y vive con ella,
#: aunque sea un parquet como los del modelo dimensional.
TEMAS = {
    "graph_sna": ("features_nodo.parquet", "banderas_rojas.csv",
                  "topologia_grafo.csv", "topologia_tripartita.csv",
                  "ego_redes_triangulos.csv", "solapamiento_actores.csv"),
    "curvas": ("curvas_roc.csv", "curvas_pr.csv", "curva_lift.csv"),
}

#: Subdirectorios que pertenecen enteros a un tema.
TEMAS_POR_CARPETA = {"v0": "modelos", "v1": "modelos", "v2": "modelos",
                     "matriz": "modelos", "gnn": "modelos"}

#: Orden en que se busca un artefacto cuyo tema no está declarado.
ORDEN_TEMAS = ("tablas", "modelos", "graph_sna", "curvas", "gold", "figuras")


def _buscar_marcador(desde):
    actual = os.path.abspath(desde)
    while True:
        if os.path.exists(os.path.join(actual, MARCADOR)):
            return actual
        padre = os.path.dirname(actual)
        if padre == actual:
            return None
        actual = padre


def raiz():
    """La raíz del repositorio.

    Se resuelve por `TFM_RAIZ` si está declarada y, si no, subiendo hasta el
    marcador.
    """
    if os.environ.get(VAR_RAIZ):
        return os.path.abspath(os.environ[VAR_RAIZ])
    encontrada = _buscar_marcador(os.path.dirname(os.path.abspath(__file__)))
    if encontrada:
        return encontrada
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def entrada():
    """Los JSONL de origen.

    No hace falta para reproducir los resultados: el procesamiento termina en
    `OUTPUTS/` y de ahí parte todo el análisis. Solo se necesita para rehacer
    la ingesta desde cero.
    """
    if os.environ.get(VAR_INPUT):
        return os.path.abspath(os.environ[VAR_INPUT])
    return os.path.join(raiz(), "INPUT")


def outputs():
    """La raíz de lo que produce el procesamiento."""
    if os.environ.get(VAR_OUTPUTS):
        return os.path.abspath(os.environ[VAR_OUTPUTS])
    return os.path.join(raiz(), "OUTPUTS")


def outputs_de(*partes):
    """Una ruta relativa a `OUTPUTS/`."""
    return os.path.join(outputs(), *partes)


def tema_de(nombre):
    """A qué tema pertenece un artefacto, por su nombre.

    Primero la carpeta, si el artefacto va dentro de una —`v1/_pred_lr.csv`—;
    después la declaración explícita de `TEMAS`; después el prefijo `tabla_`;
    y si nada de eso responde, se busca dónde está y, si no está en ningún
    sitio, se devuelve `modelos`, que es donde escriben las etapas.
    """
    nombre = str(nombre).replace("\\", "/")
    cabeza = nombre.split("/")[0]
    if cabeza in TEMAS_POR_CARPETA:
        return TEMAS_POR_CARPETA[cabeza]
    hoja = nombre.split("/")[-1]
    for t, ficheros in TEMAS.items():
        if hoja in ficheros:
            return t
    if hoja.startswith("tabla_"):
        return "tablas"
    for t in ORDEN_TEMAS:
        if os.path.exists(os.path.join(outputs(), t, nombre)):
            return t
    return "modelos"


def de(*partes):
    """Una ruta relativa a la raíz del repositorio."""
    return os.path.join(raiz(), *partes)


def entrada_de(*partes):
    """Una ruta relativa al directorio de datos de entrada."""
    return os.path.join(entrada(), *partes)


def salida(*partes, crear=True):
    """Una ruta bajo `salidas/`, creando el directorio que la contiene.

    El directorio se crea si no existe, para que un paso no dependa de que
    otro anterior lo haya creado.
    """
    ruta = os.path.join(raiz(), "salidas", *partes)
    if crear:
        os.makedirs(os.path.dirname(ruta) if os.path.splitext(ruta)[1]
                    else ruta, exist_ok=True)
    return ruta


def bronze(*partes):
    """Los nueve conjuntos de origen, ya convertidos a parquet."""
    return outputs_de("bronze", *partes)


def parquet(tabla=None):
    """El modelo dimensional; con `tabla`, el patrón de esa tabla.

    Puede resolver a la raíz que contiene a esta, por lo que dice `outputs_de`.
    """
    base = outputs_de("gold")
    return os.path.join(base, tabla, "*.parquet") if tabla else base


def parquet_de(*partes):
    """Una ruta dentro del modelo dimensional."""
    return os.path.join(outputs_de("gold"), *partes)


def referencia(*partes):
    """Los artefactos que vienen en el paquete de datos.

    Son el resultado publicado del estudio y se leen, nunca se escriben: una
    ejecución no debe alterar aquello contra lo que se compara.

    Sin argumentos devuelve la raíz de `OUTPUTS/`, porque los artefactos ya no
    viven en una sola carpeta sino repartidos por tema.
    """
    if not partes:
        return outputs()
    nombre = os.path.join(*partes).replace("\\", "/")
    return outputs_de(tema_de(nombre), nombre)


def resultados(*partes):
    """Donde escribe una ejecución sus tablas y modelos.

    Separado de `referencia()` a propósito: así se puede recalcular todo y
    contrastar lo obtenido con lo publicado, en vez de sobreescribirlo.
    """
    return salida("modelos", *partes)


def artefacto(nombre):
    """Un artefacto para leer: el recalculado si existe, si no el publicado.

    Permite ejecutar solo una parte del pipeline. Lo que se haya recalculado en
    esta sesión se usa; lo que no, se toma del paquete de datos.
    """
    propio = os.path.join(raiz(), "salidas", "modelos", nombre)
    return propio if os.path.exists(propio) else referencia(nombre)


def modelos(*partes):
    """Alias histórico de `referencia()`."""
    return referencia(*partes)


def duckdb(nombre="secop_tfm.duckdb"):
    """Un fichero DuckDB del volcado.

    Un `.duckdb` guarda rutas absolutas de la máquina que lo creó y no
    resuelve en otra. Para consultar los datos, usa `tfm.datos.conectar()`.
    """
    return outputs_de("gold", nombre)


def figuras(*partes):
    """Las figuras de esta ejecución. Las publicadas, en `OUTPUTS/figuras/`."""
    return salida("figuras", *partes)


def figuras_publicadas(*partes):
    """`OUTPUTS/figuras/`, donde escriben los módulos de `tfm.figuras`.

    El directorio se crea si no existe, para que dibujar una figura no dependa
    de que exista ya el árbol de salida.
    """
    destino = outputs_de("figuras")
    os.makedirs(destino, exist_ok=True)
    return os.path.join(destino, *partes)


def documentacion(*partes):
    """`docs/`, donde se exportan las tablas publicadas a markdown."""
    destino = os.path.join(raiz(), "docs", *partes)
    base = os.path.dirname(destino) if os.path.splitext(destino)[1] else destino
    os.makedirs(base, exist_ok=True)
    return destino


def existe_entrada():
    """Si hay algo con lo que trabajar.

    Basta con que exista alguno de los temas de `OUTPUTS/`: para reproducir
    los resultados no hace falta ni el modelo dimensional entero ni los JSONL.
    """
    return any(os.path.isdir(outputs_de(t)) for t in ORDEN_TEMAS)


def describe():
    """Resumen legible de dónde va a leer y escribir esta ejecución.

    Si los datos se están leyendo de fuera del repositorio, se dice y se dice
    por qué: es la diferencia entre una configuración deliberada y una ruta
    que nadie sabe de dónde salió.
    """
    fuera = ("   <- por la variable TFM_INPUT"
             if os.environ.get(VAR_INPUT) else "")
    fuera_out = ("   <- por la variable TFM_OUTPUTS"
                 if os.environ.get(VAR_OUTPUTS) else "")
    return "\n".join([
        f"raíz del trabajo     : {raiz()}",
        f"modelo dimensional   : {outputs_de('gold')}",
        f"JSONL de origen      : {entrada()}{fuera}",
        f"datos y resultados   : {outputs()}{fuera_out}"
        f"{'' if existe_entrada() else '   (vacío: falta descomprimir el paquete de datos)'}",
        f"salidas              : {os.path.join(raiz(), 'salidas')}",
        f"documentación        : {os.path.join(raiz(), 'docs')}",
    ])


if __name__ == "__main__":
    print(describe())
