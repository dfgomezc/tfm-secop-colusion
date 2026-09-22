#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derivación del conjunto de datos publicable.

Produce `datos/`, la versión distribuible de los artefactos del análisis, a
partir del modelo dimensional completo de `OUTPUTS/`. La diferencia entre las
dos no es de contenido analítico sino de identificadores: `OUTPUTS/` está
indexado por documento de identidad y `datos/` por seudónimo.

Qué se transforma:

- Los identificadores de actor se sustituyen por el seudónimo de
  `nodo_seudonimo`. Los representantes legales, que no figuran en esa tabla
  porque no son actores del universo contractual, reciben un seudónimo propio
  con prefijo `R-` derivado del mismo HMAC. Dos actores que comparten
  representante siguen compartiéndolo tras la sustitución, de modo que la
  estructura que el análisis explota se conserva intacta.
- Se eliminan las columnas que contienen nombre, documento, correo, teléfono o
  dirección de una persona, y las que enlazan de vuelta al registro original
  del portal, porque un identificador de proceso permite recuperar el nombre
  del actor y anula la seudonimización.
- El resto de columnas se conserva sin cambios: magnitudes, fechas, variables
  de red, puntuaciones y clasificaciones.

La correspondencia entre seudónimo y actor real no se deriva ni se escribe en
ningún fichero de `datos/`.

Uso:
    python -m tfm.datos_publicables
    python -m tfm.datos_publicables --limite-mb 20
"""

import argparse
import csv
import glob
import hashlib
import hmac
import os
import shutil

import duckdb

from tfm import rutas

#: Columnas de `nodo_2025` que no se publican. Las seis primeras identifican
#: directamente a una persona; el resto son datos de contacto o identificadores
#: de registro que permiten volver al actor real. `inscripcion_proponente` es
#: el número del Registro Único de Proponentes: no coincide con el documento
#: de identidad, pero es consultable y el análisis no lo emplea.
NODO_FUERA = (
    "documento", "nombre", "nombre_canonico", "representante_legal",
    "representante_legal_canonico", "doc_representante_legal",
    "correo", "telefono", "codigos_proveedor", "sigla", "municipio",
    "inscripcion_proponente",
)

#: Columnas de `contrato_2025` que no se publican, por el mismo criterio. El
#: objeto del contrato queda fuera porque es texto libre que con frecuencia
#: nombra a la persona contratada.
CONTRATO_FUERA = (
    "doc_rep_legal", "nombre_rep_legal", "doc_supervisor", "nombre_supervisor",
    "objeto_del_contrato", "url_proceso", "referencia_del_contrato",
    "id_portafolio",
)

#: Ficheros sin identificadores de actor: se copian tal cual.
COPIA_DIRECTA = {
    "red": ("topologia_grafo.csv", "topologia_tripartita.csv",
            "ego_redes_triangulos.csv"),
}


def sal():
    """La sal de la seudonimización, que debe existir ya.

    Se reutiliza la misma que generó `nodo_seudonimo` para que los seudónimos
    de los representantes sean estables entre ejecuciones y coherentes con los
    del resto del universo.
    """
    ruta = rutas.de("config", "sal_seudonimo.txt")
    if not os.path.exists(ruta):
        raise SystemExit(
            f"falta {ruta}; ejecuta antes `python -m tfm.identidad.seudonimizar`")
    with open(ruta, "rb") as fh:
        return fh.read().strip()


def seudonimo_representante(id_nodo, semilla):
    """Seudónimo `R-` para un representante legal.

    Se deriva del HMAC del identificador, truncado a diez dígitos hexadecimales.
    No es un correlativo porque no hay un orden que asignar: los representantes
    no forman parte del universo de actores y solo aparecen como destino de las
    aristas de representación.
    """
    digest = hmac.new(semilla, id_nodo.encode(), hashlib.sha256).hexdigest()
    return f"R-{digest[:10]}"


def columnas(con, patron):
    """Los nombres de columna de un conjunto de parquet."""
    return con.execute(f"DESCRIBE SELECT * FROM '{patron}'") \
              .df()["column_name"].tolist()


def escribir(con, consulta, destino, limite_mb):
    """Escribe el resultado de una consulta en parquet, repartido por tamaño.

    El reparto mantiene cada pieza por debajo del límite para que el conjunto
    pueda versionarse sin recurrir a almacenamiento externo. El contenido es
    idéntico al de un fichero único: todo el código lee estas carpetas con un
    comodín.
    """
    os.makedirs(destino, exist_ok=True)
    #: Las piezas de una ejecución anterior se retiran para que un reparto que
    #: pasa de varias a una no deje huérfanas, que el comodín volvería a leer.
    #: Si el sistema de ficheros no permite borrar, se avisa y se continúa:
    #: las piezas que esta ejecución escriba quedan sobreescritas de todos
    #: modos, y el aviso señala que puede haber sobrantes.
    for viejo in glob.glob(os.path.join(destino, "*.parquet")):
        try:
            os.remove(viejo)
        except OSError as exc:
            print(f"      no se pudo retirar {os.path.basename(viejo)}: {exc}")
    unico = os.path.join(destino, "part_0000.parquet")
    con.execute(f"COPY ({consulta}) TO '{unico}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")
    tam_mb = os.path.getsize(unico) / 1e6
    if tam_mb <= limite_mb:
        return [os.path.basename(unico)]

    filas = con.execute(f"SELECT count(*) FROM ({consulta})").fetchone()[0]
    trozos = int(tam_mb // limite_mb) + 1
    por_trozo = filas // trozos + 1
    nombres = []
    for i in range(trozos):
        nombre = f"part_{i:04d}.parquet"
        con.execute(
            f"COPY (SELECT * FROM ({consulta}) LIMIT {por_trozo} "
            f"OFFSET {i * por_trozo}) TO '{os.path.join(destino, nombre)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)")
        nombres.append(nombre)
    return nombres


def seudonimizar_csv(origen, destino, mapa, columna="id_nodo"):
    """Reescribe un CSV sustituyendo la columna de identificador.

    Una fila cuyo identificador no tenga seudónimo se descarta: es preferible
    perder una fila a publicar un documento de identidad.
    """
    with open(origen, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    if not filas:
        return 0
    cabecera = filas[0]
    if columna not in cabecera:
        shutil.copy2(origen, destino)
        return len(filas) - 1
    i = cabecera.index(columna)
    cabecera[i] = "seudonimo"
    salida, descartadas = [cabecera], 0
    for fila in filas[1:]:
        clave = fila[i] if i < len(fila) else ""
        if clave not in mapa:
            descartadas += 1
            continue
        fila[i] = mapa[clave]
        salida.append(fila)
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(salida)
    if descartadas:
        print(f"      {descartadas} filas sin seudónimo, descartadas")
    return len(salida) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limite-mb", type=float, default=20.0,
                    help="tamaño máximo de cada parquet (por omisión, 20)")
    a = ap.parse_args()

    semilla = sal()
    gold = rutas.parquet()
    destino_raiz = rutas.de("datos")
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")

    con.execute(f"""
        CREATE OR REPLACE VIEW seudo AS
        SELECT id_nodo, seudonimo FROM '{gold}/nodo_seudonimo/*.parquet'
    """)

    print("representantes legales sin seudónimo propio")
    reps = con.execute(f"""
        SELECT DISTINCT v.destino AS id_nodo
        FROM '{gold}/vinculo_2025/*.parquet' v
        LEFT JOIN seudo s ON v.destino = s.id_nodo
        WHERE v.tipo_vinculo = 'representado_por' AND s.id_nodo IS NULL
    """).df()
    reps["seudonimo"] = [seudonimo_representante(n, semilla)
                         for n in reps["id_nodo"]]
    con.register("reps", reps)
    con.execute("""
        CREATE OR REPLACE VIEW mapa AS
        SELECT id_nodo, seudonimo FROM seudo
        UNION ALL SELECT id_nodo, seudonimo FROM reps
    """)
    print(f"  {len(reps):,} representantes con seudónimo R-")

    # ----------------------------------------------------------- red
    print("\nvariables de red")
    origen_red = rutas.outputs_de("graph_sna")
    cols = [c for c in columnas(con, f"{origen_red}/features_nodo.parquet")
            if c != "id_nodo"]
    consulta = (f"SELECT m.seudonimo, {', '.join('f.' + c for c in cols)} "
                f"FROM '{origen_red}/features_nodo.parquet' f "
                "JOIN mapa m ON f.id_nodo = m.id_nodo "
                "ORDER BY m.seudonimo")
    piezas = escribir(con, consulta, os.path.join(destino_raiz, "red",
                                                  "features_nodo"),
                      a.limite_mb)
    print(f"  features_nodo: {len(piezas)} fichero(s)")

    mapa = dict(con.execute("SELECT id_nodo, seudonimo FROM mapa").fetchall())
    os.makedirs(os.path.join(destino_raiz, "red"), exist_ok=True)
    for nombre in ("banderas_rojas.csv", "solapamiento_actores.csv"):
        o = os.path.join(origen_red, nombre)
        if not os.path.exists(o):
            continue
        n = seudonimizar_csv(o, os.path.join(destino_raiz, "red", nombre), mapa)
        print(f"  {nombre}: {n:,} filas")
    for nombre in COPIA_DIRECTA["red"]:
        o = os.path.join(origen_red, nombre)
        if os.path.exists(o):
            shutil.copy2(o, os.path.join(destino_raiz, "red", nombre))
            print(f"  {nombre}: copiado sin cambios")

    # --------------------------------------------------------- grafo
    print("\ngrafo de 2025")
    cols_nodo = [c for c in columnas(con, f"{gold}/nodo_2025/*.parquet")
                 if c not in NODO_FUERA and c != "id_nodo"]
    consulta = (f"SELECT m.seudonimo, {', '.join('n.' + c for c in cols_nodo)} "
                f"FROM '{gold}/nodo_2025/*.parquet' n "
                "JOIN mapa m ON n.id_nodo = m.id_nodo "
                "ORDER BY m.seudonimo")
    piezas = escribir(con, consulta,
                      os.path.join(destino_raiz, "grafo", "nodo_2025"),
                      a.limite_mb)
    print(f"  nodo_2025: {len(piezas)} fichero(s), "
          f"{len(cols_nodo) + 1} columnas de {len(cols_nodo) + 1 + len(NODO_FUERA)}")

    #: El detalle de las aristas de representación es el nombre del
    #: representante. Se anula en ese tipo de arista y se conserva en los
    #: demás, donde es una categoría ('con oferta', 'telefono', 'lider').
    consulta = (
        "SELECT a.seudonimo AS origen, b.seudonimo AS destino, "
        "v.tipo_vinculo, v.peso, v.valor, "
        "CASE WHEN v.tipo_vinculo = 'representado_por' THEN NULL "
        "     ELSE v.detalle END AS detalle "
        f"FROM '{gold}/vinculo_2025/*.parquet' v "
        "JOIN mapa a ON v.origen = a.id_nodo "
        "JOIN mapa b ON v.destino = b.id_nodo "
        "ORDER BY v.tipo_vinculo, a.seudonimo, b.seudonimo")
    piezas = escribir(con, consulta,
                      os.path.join(destino_raiz, "grafo", "vinculo_2025"),
                      a.limite_mb)
    print(f"  vinculo_2025: {len(piezas)} fichero(s)")

    # --------------------------------------------------------- curvas
    print("\ncurvas")
    os.makedirs(os.path.join(destino_raiz, "curvas"), exist_ok=True)
    for o in sorted(glob.glob(os.path.join(rutas.outputs_de("curvas"), "*.csv"))):
        shutil.copy2(o, os.path.join(destino_raiz, "curvas",
                                     os.path.basename(o)))
        print(f"  {os.path.basename(o)}")

    # --------------------------------------------------------- tablas
    print("\ntablas publicadas")
    os.makedirs(os.path.join(destino_raiz, "tablas"), exist_ok=True)
    for o in sorted(glob.glob(os.path.join(rutas.outputs_de("tablas"), "*.csv"))):
        shutil.copy2(o, os.path.join(destino_raiz, "tablas",
                                     os.path.basename(o)))
    print(f"  {len(glob.glob(os.path.join(destino_raiz, 'tablas', '*.csv')))} tablas")

    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(destino_raiz) for f in fs)
    print(f"\n{destino_raiz}: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
