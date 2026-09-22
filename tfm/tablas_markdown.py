#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exportación de las tablas publicadas a markdown.

Convierte cada `OUTPUTS/tablas/tabla_*.csv` en un fichero markdown bajo
`docs/tablas/`, con las cifras en convención decimal española y las columnas
numéricas alineadas a la derecha. El objetivo es que las tablas del estudio
puedan leerse sin abrir un CSV y sin recalcular nada: son las mismas cifras,
el mismo orden y los mismos encabezados que produce el análisis.

Cada fichero declara el CSV del que procede, de modo que el origen de cualquier
cifra es verificable a partir del propio documento.

Uso:
    python -m tfm.tablas_markdown
    python -m tfm.tablas_markdown --tabla tabla_5_18_variantes
"""

import argparse
import csv
import glob
import os
import re

from tfm import rutas

#: Título de cada tabla. La clave es el nombre del CSV sin extensión. Las que
#: no aparecen aquí se titulan a partir del nombre del fichero.
TITULOS = {
    "tabla_5_9_logistica":
        "Rejilla de hiperparámetros de la regresión logística",
    "tabla_5_10_random_forest":
        "Rejilla de hiperparámetros del bosque aleatorio",
    "tabla_5_11_gradient_boosting":
        "Rejilla de hiperparámetros del gradient boosting",
    "tabla_5_12_gnn":
        "Rejilla de hiperparámetros del perceptrón multicapa sobre variables "
        "propagadas",
    "tabla_5_13_comparativa":
        "Comparativa de los cuatro modelos sobre el mismo conjunto de prueba",
    "tabla_5_13_proveedores_cri":
        "Proveedores con mayor puntuación en el índice de riesgo",
    "tabla_5_14_entidades_triangulos":
        "Entidades con mayor número de triángulos de riesgo",
    "tabla_5_15_pares_coincidencia":
        "Pares de actores con mayor coincidencia en procesos",
    "tabla_5_16_fuga_magnitudes":
        "Magnitudes del control de fuga entre la etiqueta y las variables",
    "tabla_5_17_fuga_cri":
        "Reconstruibilidad de cada componente del índice desde las variables",
    "tabla_5_18_variantes":
        "Comparativa bajo las tres variantes de control",
    "tabla_5_19_matriz":
        "Matriz de control: cuatro arquitecturas por dos conjuntos de entrada",
    "tabla_5_20_brazo_clasico":
        "Los tres ordenamientos que no entrenan nada",
    "tabla_5_21_siete_ordenamientos":
        "Los siete ordenamientos evaluados sobre el mismo conjunto de prueba",
    "tabla_5_22_banderas_diagnostico":
        "Diagnóstico de cada bandera roja estructural por separado",
    "tabla_5_23_solapamiento":
        "Solapamiento entre la lista corta del modelo y la señal estructural",
    "tabla_5_24_zonas":
        "Perfil mediano de cada zona de solapamiento",
    "tabla_5_25_precision_k":
        "Precisión y lift según el tamaño de la lista corta",
    "tabla_5_26_umbral_f1":
        "F1 al umbral convencional y al umbral que lo maximiza",
    "tabla_5_27_confusion":
        "Matriz de confusión del modelo relacional a los dos umbrales",
    "tabla_5_28_calibracion":
        "Puntuación de Brier y diagrama de fiabilidad",
    "tabla_5_29_nueve_ordenamientos":
        "Los nueve ordenamientos evaluados sobre el mismo conjunto de prueba",
    "tabla_5_30_explicabilidad":
        "Explicación de las diez alertas de mayor puntuación",
    "tabla_5_31_gcn_vs_sgc":
        "La red convolucional de grafos y el modelo relacional que la "
        "simplifica",
    "tabla_5_32_convergencia_gcn":
        "Trayectoria de validación de la GCN en los hitos del ajuste",
    "tabla_5_33_explicabilidad_comparada":
        "Las dos explicaciones, sobre la misma partición y las mismas "
        "variables",
    "tabla_5_34_incertidumbre":
        "Intervalo de confianza al 95 % del AUC-ROC, por remuestreo",
    "tabla_5_35_comparaciones":
        "Diferencia emparejada de AUC-ROC entre pares de modelos",
    "tabla_5_x_sensibilidad_exclusion":
        "Sensibilidad del recuento de triángulos al criterio de exclusión",
    "tabla_5_x_atribucion_decil":
        "Atribución al entorno por decil de puntuación",
}

#: Columnas cuyo contenido es un rótulo aunque parezca numérico. Se alinean a
#: la izquierda para que la tabla se lea como una lista y no como una serie.
ROTULOS = {"Variante", "Umbral", "Decil", "Tipo", "Hito", "tramo"}


def es_numero(texto):
    """Si un valor de celda es una cifra, en cualquiera de las dos notaciones.

    Acepta tanto el punto decimal del CSV como la coma que ya traen algunas
    columnas, y tolera el signo, la notación científica y el separador de
    millar.
    """
    limpio = texto.strip().replace("−", "-")
    if not limpio or limpio in {"-", "—"}:
        return False
    return bool(re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?"
                             r"(?:[eE][+-]?\d+)?", limpio))


def a_decimal_espanol(texto):
    """Pasa una cifra de notación inglesa a española.

    El punto decimal se convierte en coma y el separador de millar en punto.
    Un valor que no sea una cifra se devuelve intacto.
    """
    limpio = texto.strip()
    if not es_numero(limpio) or "," in limpio:
        return limpio
    if "e" in limpio.lower():
        return limpio.replace(".", ",")
    entera, _, decimal = limpio.partition(".")
    signo = ""
    if entera[:1] in "+-":
        signo, entera = entera[0], entera[1:]
    if len(entera) > 4:
        grupos = []
        while len(entera) > 3:
            grupos.insert(0, entera[-3:])
            entera = entera[:-3]
        grupos.insert(0, entera)
        entera = ".".join(grupos)
    return signo + entera + (f",{decimal}" if decimal else "")


def titulo_de(nombre):
    """El título de una tabla, del catálogo o derivado de su nombre."""
    if nombre in TITULOS:
        return TITULOS[nombre]
    resto = re.sub(r"^tabla_[0-9x_]+_", "", nombre).replace("_", " ")
    return resto.capitalize()


def leer(ruta):
    """Devuelve el encabezado y las filas de un CSV."""
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    if not filas:
        return [], []
    return filas[0], filas[1:]


def a_markdown(nombre, cabecera, filas):
    """Arma el documento markdown de una tabla.

    Las columnas se alinean a la derecha cuando todas sus celdas con contenido
    son cifras, y a la izquierda en cualquier otro caso.
    """
    numerica = []
    for i, col in enumerate(cabecera):
        valores = [f[i] for f in filas if i < len(f) and f[i].strip()]
        numerica.append(bool(valores) and col not in ROTULOS
                        and all(es_numero(v) for v in valores))

    lineas = [
        f"# {titulo_de(nombre)}",
        "",
        f"Fuente: `OUTPUTS/tablas/{nombre}.csv`, generado por el análisis.",
        "",
        "| " + " | ".join(cabecera) + " |",
        "|" + "|".join("---:" if n else ":---" for n in numerica) + "|",
    ]
    for fila in filas:
        celdas = [(fila[i] if i < len(fila) else "") for i in range(len(cabecera))]
        lineas.append("| " + " | ".join(
            a_decimal_espanol(c) if numerica[i] else c.strip()
            for i, c in enumerate(celdas)) + " |")
    lineas.append("")
    return "\n".join(lineas)


def indice(nombres):
    """Arma el índice de `docs/tablas/README.md`."""
    lineas = [
        "# Tablas publicadas",
        "",
        "Las tablas del análisis, exportadas desde los CSV que produce el",
        "código. Se regeneran con `python -m tfm.tablas_markdown`.",
        "",
        "| Tabla | Contenido |",
        "|:---|:---|",
    ]
    for n in nombres:
        lineas.append(f"| [`{n}`]({n}.md) | {titulo_de(n)} |")
    lineas.append("")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tabla", help="exporta solo esta tabla, sin extensión")
    a = ap.parse_args()

    origen = rutas.outputs_de("tablas")
    patron = f"{a.tabla}.csv" if a.tabla else "tabla_*.csv"
    rutas_csv = sorted(glob.glob(os.path.join(origen, patron)))
    if not rutas_csv:
        raise SystemExit(f"no hay tablas que exportar en {origen}")

    exportadas = []
    for ruta_csv in rutas_csv:
        nombre = os.path.splitext(os.path.basename(ruta_csv))[0]
        cabecera, filas = leer(ruta_csv)
        if not cabecera:
            print(f"  {nombre}: vacío, se omite")
            continue
        destino = rutas.documentacion("tablas", f"{nombre}.md")
        with open(destino, "w", encoding="utf-8") as fh:
            fh.write(a_markdown(nombre, cabecera, filas))
        exportadas.append(nombre)
        print(f"  {nombre}.md  ({len(filas)} filas)")

    if not a.tabla and exportadas:
        with open(rutas.documentacion("tablas", "README.md"),
                  "w", encoding="utf-8") as fh:
            fh.write(indice(exportadas))
        print(f"  README.md  (índice de {len(exportadas)} tablas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
