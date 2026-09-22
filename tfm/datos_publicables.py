#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derivación del conjunto de datos publicable.

Produce `datos/`, la versión distribuible de los artefactos del análisis, a
partir del árbol completo de `OUTPUTS/`. La diferencia entre los dos no es de
contenido analítico sino de identificadores: `OUTPUTS/` está indexado por
documento de identidad y `datos/` por seudónimo.

El esquema se conserva. Los nombres de columna, `id_nodo` incluido, son los
mismos, y lo que cambia es el valor. Y la estructura de directorios reproduce
la de `OUTPUTS/`, de modo que el conjunto funciona sin adaptar nada:

    TFM_OUTPUTS=datos python -m tfm.modelos.comparativa
    TFM_OUTPUTS=datos python -m tfm.figuras.generales

Qué se transforma:

- Los identificadores de actor se sustituyen por el seudónimo de
  `nodo_seudonimo`. Los representantes legales, que no figuran en esa tabla
  porque no son actores del universo contractual, reciben un seudónimo propio
  con prefijo `R-` derivado del mismo HMAC. Dos actores que comparten
  representante siguen compartiéndolo tras la sustitución, de modo que la
  estructura que el análisis explota se conserva intacta.
- Los identificadores de proceso y de contrato se sustituyen por un resumen
  con sal, porque permiten recuperar el expediente en el portal y con él el
  nombre del actor. Se sustituyen en lugar de eliminarse porque hay uniones
  del análisis que los necesitan como clave.
- Se eliminan las columnas que contienen nombre, documento, correo, teléfono o
  municipio de una persona, el objeto del contrato, que es texto libre donde
  con frecuencia aparece el nombre de la persona contratada, y el número del
  Registro Único de Proponentes, que es consultable.
- `fact_contrato` se reduce a la única columna que el análisis le pide, la
  fecha de firma, con la que se obtiene el recuento del universo de partida.
- El resto se conserva sin cambios: magnitudes, fechas, variables de red,
  predicciones, atribuciones, puntuaciones y clasificaciones.

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
import pandas as pd

from tfm import rutas
from tfm.datos import CIIU_FINANCIERO, PATRON_FINANCIERO

#: Columnas de las tablas de actores que no se publican. Identifican
#: directamente a una persona, son datos de contacto, o son identificadores de
#: registro que permiten volver al actor real. `inscripcion_proponente` es el
#: número del Registro Único de Proponentes: no coincide con el documento de
#: identidad, pero es consultable y el análisis no lo emplea.
NODO_FUERA = (
    "documento", "nombre", "nombre_canonico", "representante_legal",
    "representante_legal_canonico", "doc_representante_legal",
    "correo", "telefono", "codigos_proveedor", "sigla", "municipio",
    "inscripcion_proponente", "ciudad",
)

#: Columnas de `contrato_2025` que no se publican. El objeto del contrato queda
#: fuera porque es texto libre que con frecuencia nombra a la persona
#: contratada, y la URL del proceso porque lleva al expediente en el portal.
CONTRATO_FUERA = (
    "doc_rep_legal", "nombre_rep_legal", "doc_supervisor", "nombre_supervisor",
    "objeto_del_contrato", "url_proceso", "referencia_del_contrato",
)

#: Identificadores de proceso, de contrato y de cuenta de plataforma. No
#: designan a una persona, pero permiten recuperar el expediente o el perfil
#: del proveedor en el portal, de modo que se sustituyen por un resumen con sal
#: en lugar de eliminarse: el análisis une por ellos y los recuentos de cuentas
#: distintas, que no coinciden con los de actores, dependen de que sigan siendo
#: distintos entre sí. El resumen es biyectivo, así que los recuentos no
#: cambian.
CLAVES_OPACAS = ("id_contrato", "id_portafolio", "codigo_proveedor",
                 "codigo_entidad")

#: Ficheros de resultados sin identificador de actor. Se copian tal cual.
COPIA_DIRECTA = (
    "graph_sna/topologia_grafo.csv",
    "graph_sna/topologia_tripartita.csv",
    "graph_sna/ego_redes_triangulos.csv",
)

#: Subárboles de `OUTPUTS/modelos/` que se publican enteros. Las predicciones
#: de las variantes son pares (etiqueta, probabilidad) sin identificador, y las
#: rejillas son combinaciones de hiperparámetros con su métrica.
MODELOS_DIRECTOS = ("v0", "v1", "v2", "matriz")

#: Carpetas de trabajo que no se publican, en cualquier nivel.
CARPETAS_EXCLUIDAS = {"_to_delete", "__pycache__", ".ipynb_checkpoints"}


def sal():
    """La sal de la seudonimización, que debe existir ya.

    Se reutiliza la misma que generó `nodo_seudonimo` para que los seudónimos
    derivados aquí sean estables entre ejecuciones y coherentes con los del
    resto del universo.
    """
    ruta = rutas.de("config", "sal_seudonimo.txt")
    if not os.path.exists(ruta):
        raise SystemExit(
            f"falta {ruta}; ejecuta antes `python -m tfm.identidad.seudonimizar`")
    with open(ruta, "rb") as fh:
        return fh.read().strip()


def opaco(valor, semilla, prefijo=""):
    """Resumen con sal de un identificador, truncado a doce hexadecimales."""
    if valor is None:
        return None
    digest = hmac.new(semilla, str(valor).encode(), hashlib.sha256).hexdigest()
    return f"{prefijo}{digest[:12]}"


def columnas(con, patron):
    """Los nombres de columna de un conjunto de parquet."""
    return con.execute(f"DESCRIBE SELECT * FROM '{patron}'") \
              .df()["column_name"].tolist()


def escribir_parquet(con, consulta, destino, limite_mb):
    """Escribe el resultado de una consulta en parquet, repartido por tamaño.

    El reparto mantiene cada pieza por debajo del límite para que el conjunto
    pueda versionarse sin almacenamiento externo. El contenido es idéntico al
    de un fichero único: el código lee estas carpetas con un comodín, de modo
    que el motor junta las partes al vuelo.
    """
    os.makedirs(destino, exist_ok=True)
    #: Las piezas de una ejecución anterior se retiran para que un reparto que
    #: pasa de varias a una no deje huérfanas, que el comodín volvería a leer.
    #: Si el sistema de ficheros no permite borrar, se avisa y se continúa.
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
        return 1

    filas = con.execute(f"SELECT count(*) FROM ({consulta})").fetchone()[0]
    trozos = int(tam_mb // limite_mb) + 1
    por_trozo = filas // trozos + 1
    for i in range(trozos):
        con.execute(
            f"COPY (SELECT * FROM ({consulta}) LIMIT {por_trozo} "
            f"OFFSET {i * por_trozo}) "
            f"TO '{os.path.join(destino, f'part_{i:04d}.parquet')}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)")
    return trozos


def reescribir_csv(origen, destino, mapa, columnas_actor=("id_nodo",),
                   columnas_fuera=()):
    """Copia un CSV sustituyendo por seudónimo las columnas de actor.

    Los nombres de columna se conservan salvo los que se retiran a proposito:
    lo que cambia es el valor. Una fila cuyo identificador no tenga seudónimo
    se descarta, porque es preferible perder una fila a publicar un documento
    de identidad.
    """
    with open(origen, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    if not filas:
        return 0
    cabecera = filas[0]
    idx_actor = [i for i, c in enumerate(cabecera) if c in columnas_actor]
    idx_fuera = {i for i, c in enumerate(cabecera) if c in columnas_fuera}
    salida = [[c for i, c in enumerate(cabecera) if i not in idx_fuera]]
    descartadas = 0
    for fila in filas[1:]:
        perdida = False
        for i in idx_actor:
            if i >= len(fila) or not fila[i]:
                continue
            if fila[i] in mapa:
                fila[i] = mapa[fila[i]]
            elif i not in idx_fuera:
                perdida = True
        if perdida:
            descartadas += 1
            continue
        salida.append([c for i, c in enumerate(fila) if i not in idx_fuera])
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(salida)
    if descartadas:
        print(f"      {descartadas} filas sin seudónimo, descartadas")
    return len(salida) - 1


def copiar_arbol(origen, destino):
    """Copia un subárbol omitiendo las carpetas de trabajo."""
    n = 0
    for raiz_dir, dirs, ficheros in os.walk(origen):
        dirs[:] = [d for d in dirs if d not in CARPETAS_EXCLUIDAS]
        rel = os.path.relpath(raiz_dir, origen)
        sal_dir = destino if rel == "." else os.path.join(destino, rel)
        os.makedirs(sal_dir, exist_ok=True)
        for f in ficheros:
            shutil.copy2(os.path.join(raiz_dir, f), os.path.join(sal_dir, f))
            n += 1
    return n


def publicar_gold(con, gold, salida, tabla, limite_mb, fuera=(), actor=None,
                  extra=()):
    """Publica una tabla del modelo dimensional con el esquema conservado.

    `extra` son pares (expresión, alias) que se añaden como columnas nuevas.
    Sirven para materializar un criterio que depende de una columna que no se
    publica, de modo que el análisis pueda aplicarlo sin ella.
    """
    cols = columnas(con, f"{gold}/{tabla}/*.parquet")
    piezas = []
    for c in cols:
        if c in fuera:
            continue
        if c == actor:
            piezas.append(f"m.seudonimo AS {c}")
        elif c in CLAVES_OPACAS:
            piezas.append(f"opacar(CAST(t.{c} AS VARCHAR)) AS {c}")
        else:
            piezas.append(f"t.{c}")
    for expresion, alias in extra:
        piezas.append(f"{expresion} AS {alias}")
    unir = f"JOIN mapa m ON t.{actor} = m.id_nodo" if actor else ""
    consulta = (f"SELECT {', '.join(piezas)} "
                f"FROM '{gold}/{tabla}/*.parquet' t {unir} ORDER BY 1")
    n = escribir_parquet(con, consulta,
                         os.path.join(salida, "gold", tabla), limite_mb)
    print(f"  {tabla}: {n} fichero(s), {len(piezas)} columnas de {len(cols)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limite-mb", type=float, default=20.0,
                    help="tamaño máximo de cada parquet (por omisión, 20)")
    a = ap.parse_args()

    semilla = sal()
    gold = rutas.parquet()
    salida = rutas.de("datos")
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")
    con.create_function("opacar", lambda v: opaco(v, semilla),
                        ["VARCHAR"], "VARCHAR")

    # ------------------------------------------- mapa de seudónimos
    con.execute(f"""
        CREATE OR REPLACE VIEW seudo AS
        SELECT id_nodo, seudonimo FROM '{gold}/nodo_seudonimo/*.parquet'
    """)
    reps = con.execute(f"""
        SELECT DISTINCT v.destino AS id_nodo
        FROM '{gold}/vinculo_2025/*.parquet' v
        LEFT JOIN seudo s ON v.destino = s.id_nodo
        WHERE v.tipo_vinculo = 'representado_por' AND s.id_nodo IS NULL
    """).df()
    reps["seudonimo"] = [opaco(n, semilla, "R-") for n in reps["id_nodo"]]
    con.register("reps", reps)
    con.execute("""
        CREATE OR REPLACE VIEW mapa AS
        SELECT id_nodo, seudonimo FROM seudo
        UNION ALL SELECT id_nodo, seudonimo FROM reps
    """)
    mapa = dict(con.execute("SELECT id_nodo, seudonimo FROM mapa").fetchall())
    print(f"mapa de seudónimos: {len(mapa):,} actores "
          f"({len(reps):,} representantes con prefijo R-)")

    # ---------------------------------------------------- gold
    print("\nmodelo dimensional")
    #: `nodo_2025` se publica con una columna añadida: el criterio de banca y
    #: seguros ya evaluado. Depende de la razón social, que no se publica, de
    #: modo que sin materializarlo aquí las etapas que excluyen al sector no
    #: podrían ejecutarse sobre el conjunto distribuido.
    financiero = " OR ".join(
        f"substr(t.ciiu1, 1, 2) = '{d}'" for d in CIIU_FINANCIERO)
    criterio = (f"((t.ciiu1 IS NOT NULL AND ({financiero})) "
                f"OR regexp_matches(upper(t.nombre), '{PATRON_FINANCIERO}'))")
    publicar_gold(con, gold, salida, "nodo_2025", a.limite_mb,
                  NODO_FUERA, "id_nodo",
                  extra=[(criterio, "es_banca_seguros")])
    publicar_gold(con, gold, salida, "nodo_entidad", a.limite_mb,
                  NODO_FUERA, "id_nodo")
    #: En el conjunto publicado el identificador del actor ya es su seudónimo,
    #: de modo que la tabla de correspondencia se convierte en la identidad.
    #: Se conserva porque hay etapas que unen por ella, y se escribe con su
    #: nombre de fichero original, que es el que esas etapas piden.
    destino_seudo = os.path.join(salida, "gold", "nodo_seudonimo")
    os.makedirs(destino_seudo, exist_ok=True)
    con.execute(
        "COPY (SELECT s.seudonimo AS id_nodo, s.tipo_nodo, s.seudonimo "
        f"FROM '{gold}/nodo_seudonimo/*.parquet' s ORDER BY 1) "
        f"TO '{destino_seudo}/nodo_seudonimo.parquet' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)")
    print("  nodo_seudonimo: identidad, con el nombre de fichero original")
    publicar_gold(con, gold, salida, "contrato_2025", a.limite_mb,
                  CONTRATO_FUERA)
    publicar_gold(con, gold, salida, "proceso_competencia", a.limite_mb)
    #: `nodo_cuenta` es el puente entre la cuenta de plataforma y el actor
    #: resuelto. Sin ella no se puede comprobar que el número de cuentas y el
    #: de actores son magnitudes distintas, que es uno de los invariantes.
    publicar_gold(con, gold, salida, "nodo_cuenta", a.limite_mb,
                  NODO_FUERA, "id_nodo")
    publicar_gold(con, gold, salida, "splink_parametros", a.limite_mb)

    #: El detalle de las aristas de representación es el nombre del
    #: representante. Se anula en ese tipo de arista y se conserva en los
    #: demás, donde es una categoría: 'con oferta', 'telefono', 'lider'.
    consulta = ("SELECT a.seudonimo AS origen, b.seudonimo AS destino, "
                "v.tipo_vinculo, v.peso, v.valor, "
                "CASE WHEN v.tipo_vinculo = 'representado_por' THEN NULL "
                "     ELSE v.detalle END AS detalle "
                f"FROM '{gold}/vinculo_2025/*.parquet' v "
                "JOIN mapa a ON v.origen = a.id_nodo "
                "JOIN mapa b ON v.destino = b.id_nodo "
                "ORDER BY v.tipo_vinculo, a.seudonimo, b.seudonimo")
    n = escribir_parquet(con, consulta,
                         os.path.join(salida, "gold", "vinculo_2025"),
                         a.limite_mb)
    print(f"  vinculo_2025: {n} fichero(s)")

    #: De `fact_contrato` el análisis solo pide el recuento de contratos
    #: firmados en 2025, que es la primera barra del embudo de alcance. Se
    #: publica reducida a la fecha de firma: la consulta funciona sin cambios y
    #: no viajan 656 MB ni un solo identificador.
    consulta = (f"SELECT fecha_firma FROM '{gold}/fact_contrato/*.parquet' "
                "WHERE fecha_firma BETWEEN DATE '2025-01-01' "
                "AND DATE '2025-12-31' ORDER BY fecha_firma")
    n = escribir_parquet(con, consulta,
                         os.path.join(salida, "gold", "fact_contrato"),
                         a.limite_mb)
    print(f"  fact_contrato: {n} fichero(s), reducida a la fecha de firma")

    # ----------------------------------------------- graph_sna
    print("\nvariables de red")
    origen = rutas.outputs_de("graph_sna")
    destino = os.path.join(salida, "graph_sna")
    os.makedirs(destino, exist_ok=True)
    #: El orden físico de las filas se conserva exactamente. No es un detalle
    #: estético: `train_test_split` permuta índices posicionales, de modo que
    #: reordenar el fichero cambia el conjunto de prueba y con él todas las
    #: métricas. El fichero de origen no está ordenado por ninguna columna,
    #: así que se copia fila a fila en lugar de reconstruirlo con una consulta,
    #: que no garantiza el orden.
    marco = pd.read_parquet(f"{origen}/features_nodo.parquet")
    sin_seudonimo = ~marco["id_nodo"].isin(mapa)
    if sin_seudonimo.any():
        raise SystemExit(
            f"{int(sin_seudonimo.sum())} filas de features_nodo sin seudónimo")
    marco["id_nodo"] = marco["id_nodo"].map(mapa)
    marco.to_parquet(f"{destino}/features_nodo.parquet",
                     index=False, compression="zstd")
    print(f"  features_nodo.parquet: {len(marco.columns)} columnas, "
          f"{len(marco):,} filas en su orden original")

    for nombre in ("banderas_rojas.csv", "solapamiento_actores.csv"):
        o = os.path.join(origen, nombre)
        if os.path.exists(o):
            print(f"  {nombre}: "
                  f"{reescribir_csv(o, os.path.join(destino, nombre), mapa):,} filas")
    for rel in COPIA_DIRECTA:
        o = rutas.outputs_de(*rel.split("/"))
        if os.path.exists(o):
            shutil.copy2(o, os.path.join(salida, rel))
            print(f"  {os.path.basename(rel)}: copiado sin cambios")

    # ------------------------------------------------- modelos
    print("\nartefactos de los modelos")
    om = rutas.outputs_de("modelos")
    dm = os.path.join(salida, "modelos")
    os.makedirs(dm, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(om, "*.csv"))):
        base = os.path.basename(f)
        if base == "particion_prueba.csv":
            print(f"  {base}: {reescribir_csv(f, os.path.join(dm, base), mapa):,} filas")
        else:
            shutil.copy2(f, os.path.join(dm, base))
    print(f"  {len(glob.glob(os.path.join(dm, '*.csv')))} CSV en la raíz")

    for sub in MODELOS_DIRECTOS:
        o = os.path.join(om, sub)
        if os.path.isdir(o):
            print(f"  {sub}/: {copiar_arbol(o, os.path.join(dm, sub))} ficheros")

    #: Las predicciones de las dos redes con paso de mensajes llevan
    #: identificador; las historias de entrenamiento y los puntos de control
    #: son series de pérdida y matrices de pesos.
    o = os.path.join(om, "gnn")
    if os.path.isdir(o):
        dg = os.path.join(dm, "gnn")
        os.makedirs(dg, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(o, "*"))):
            base = os.path.basename(f)
            if not os.path.isfile(f):
                continue
            if base.startswith("_pred_") and base.endswith(".csv"):
                reescribir_csv(f, os.path.join(dg, base), mapa)
            else:
                shutil.copy2(f, os.path.join(dg, base))
        print(f"  gnn/: {len(os.listdir(dg))} ficheros")

    #: Las tablas de explicabilidad traen el identificador y el seudónimo en
    #: columnas separadas. Se retiran las del identificador y se conservan las
    #: del seudónimo, que son las que el documento cita.
    o = os.path.join(om, "explicabilidad")
    if os.path.isdir(o):
        de = os.path.join(dm, "explicabilidad")
        for raiz_dir, dirs, ficheros in os.walk(o):
            dirs[:] = [d for d in dirs if d not in CARPETAS_EXCLUIDAS]
            rel = os.path.relpath(raiz_dir, o)
            sal_dir = de if rel == "." else os.path.join(de, rel)
            os.makedirs(sal_dir, exist_ok=True)
            for f in ficheros:
                ro, rd = os.path.join(raiz_dir, f), os.path.join(sal_dir, f)
                if f.endswith(".csv"):
                    reescribir_csv(ro, rd, mapa,
                                   columnas_actor=("id_nodo", "vecino"),
                                   columnas_fuera=("id_nodo", "vecino"))
                else:
                    shutil.copy2(ro, rd)
        print(f"  explicabilidad/: "
              f"{sum(len(f) for _, _, f in os.walk(de))} ficheros")

    # --------------------------------- curvas, tablas y figuras
    print("\nresultados publicados")
    for tema in ("curvas", "tablas", "figuras"):
        o = rutas.outputs_de(tema)
        if os.path.isdir(o):
            print(f"  {tema}/: {copiar_arbol(o, os.path.join(salida, tema))} ficheros")

    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(salida) for f in fs)
    print(f"\n{salida}: {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
