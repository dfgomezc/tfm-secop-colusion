#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tablas de actores sin nombres propios.

Regenera desde los artefactos las tablas de actores que asocian una entidad o
una empresa concreta a su puntuación de riesgo, sustituyendo el nombre por el
seudónimo de `nodo_seudonimo` y conservando la descripción cualitativa, que es
lo que da valor al resultado: naturaleza jurídica del actor, departamento y
magnitudes.

Produce además la clave de correspondencia limitada a los actores citados, unas
decenas frente a las 143.420 del universo. Esa clave contiene datos de carácter
personal y queda fuera del control de versiones.

Salidas:
    OUTPUTS/tabla_5_13_proveedores_cri.csv
    OUTPUTS/tabla_5_14_entidades_triangulos.csv
    OUTPUTS/tabla_5_15_pares_coincidencia.csv
    salidas/clave_seudonimos_RESTRINGIDO.csv

Uso:
    python -m tfm.modelos.tablas_seudonimizadas
"""

import os
import re

import duckdb
import pandas as pd

from tfm import rutas
from tfm.grafo import triangulos

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
MODELOS = rutas.resultados()
#: Se resuelve con `artefacto` y no contra `salidas/modelos/`: así el módulo
#: corre sobre un clon en el que solo está el paquete de datos, sin obligar a
#: rehacer antes la etapa de grafo. Si hay un recálculo propio, `artefacto` lo
#: prefiere; si no, toma el publicado.
FEATURES = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
SALIDAS = os.path.join(rutas.salida())

CIIU_FINANCIERO = ("64", "65", "66")
PATRON_FINANCIERO = ("SEGURO|ASEGURADORA|FIDUCIARIA|BANCO|"
                     "COMPANIA DE SEGUROS|CORREDORES DE SEGUROS")

#: Familias de actor que el documento describe cualitativamente. La descripción
#: sustituye al nombre: dice de qué clase de organismo se trata sin decir cuál.
FAMILIAS = [
    #: El sector asegurador va primero: sus uniones temporales llevan a menudo
    #: el nombre del municipio contratante y se clasificarían como alcaldía.
    (r"SEGURO|ASEGURADORA|FIDUCIARIA|BANCO|PREVISORA|POSITIVA",
     "Sector asegurador"),
    #: Los entes descentralizados municipales aparecen a menudo con siglas
    #: —EDUR, EDURHA, EDUOCCIDENTE— y no con su denominación completa.
    (r"EMPRESA DE DESARROLLO|DESARROLLO URBANO|DESARROLLO TERRITORIAL"
     r"|^EDU|EMPRESA AUTONOMA|EMPRESA AUTÓNOMA",
     "Empresa municipal de desarrollo"),
    (r"FONDO DE DESARROLLO|FONDO DE VIVIENDA|FONDO MIXTO", "Fondo de desarrollo"),
    (r"CORPORACION|CORPORACIÓN", "Corporación de economía mixta"),
    (r"E\.?S\.?E\.?|HOSPITAL|SALUD", "Entidad prestadora de salud"),
    (r"ALCALDIA|ALCALDÍA|MUNICIPIO", "Alcaldía municipal"),
    (r"GOBERNACION|GOBERNACIÓN|DEPARTAMENTO DE|DEPARTAMENTO DEL",
     "Gobernación departamental"),
    (r"AERONAUTICA|AEROCIVIL", "Entidad nacional del sector transporte"),
    (r"ICBF|BIENESTAR FAMILIAR", "Entidad nacional del sector social"),
    (r"COLOMBIA COMPRA", "Agencia nacional de contratación"),
    (r"UNIVERSIDAD|INSTITUTO TECNOL", "Institución de educación superior"),
    (r"\bUT\b|UNION TEMPORAL|UNIÓN TEMPORAL|CONSORCIO", "Unión temporal"),
    (r"INGENIERIA|INGENIERÍA|CONSTRUCTORA|OBRAS", "Empresa de ingeniería y obra"),
    (r"SERVICIOS|SUMINISTROS|COMERCIALIZADORA", "Empresa de servicios y suministros"),
]


def familia(nombre, tipo=None):
    """Clase de organismo, para conservar la lectura sin el nombre propio.

    El tipo de nodo manda sobre el nombre cuando ambos están disponibles: una
    unión temporal suele llamarse como el objeto que contrata, no como lo que
    es, y clasificarla por el nombre da resultados equivocados.
    """
    if not nombre:
        return "Unión temporal" if tipo == "union_temporal" else "Actor sin clasificar"
    n = nombre.upper()
    if tipo == "union_temporal" and not re.search(
            r"SEGURO|ASEGURADORA|PREVISORA|POSITIVA", n):
        return "Unión temporal"
    for patron, etiqueta in FAMILIAS:
        if re.search(patron, n):
            return etiqueta
    return "Empresa privada"


def esp(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    seudo = os.path.join(PQ, "nodo_seudonimo", "nodo_seudonimo.parquet")
    if not os.path.exists(seudo):
        raise SystemExit("falta nodo_seudonimo; ejecuta antes 10_seudonimizar.py")

    con.execute(f"""
        CREATE OR REPLACE VIEW nodo AS SELECT * FROM '{PQ}/nodo_2025/*.parquet';
        --: Las entidades contratantes no están en nodo_2025 sino en su propia
        --: tabla, con sector y orden administrativo, que es lo que permite
        --: describirlas sin nombrarlas.
        CREATE OR REPLACE VIEW ent AS SELECT * FROM '{PQ}/nodo_entidad/*.parquet';
        CREATE OR REPLACE VIEW v AS SELECT * FROM '{PQ}/vinculo_2025/*.parquet';
        CREATE OR REPLACE VIEW s AS SELECT * FROM '{seudo}';
        CREATE OR REPLACE VIEW fx AS
          SELECT * FROM '{FEATURES}';
    """)
    citados = []

    # ---- Tabla 5.13: proveedores de mayor CRI --------------------------
    t13 = con.execute("""
        SELECT s.seudonimo, n.nombre, n.departamento_norm AS depto,
               f.cri_score, f.n_contratos, f.n_entidades,
               f.pct_single, f.pct_directa
        FROM fx f JOIN nodo n USING (id_nodo) JOIN s USING (id_nodo)
        WHERE f.n_contratos >= 40
        ORDER BY f.cri_score DESC LIMIT 8
    """).df()
    t13["familia"] = t13["nombre"].map(familia)  # todos son proveedores
    citados += list(zip(t13["seudonimo"], t13["nombre"]))
    salida13 = t13[["seudonimo", "familia", "depto", "cri_score", "n_contratos",
                    "n_entidades", "pct_single", "pct_directa"]]
    salida13.to_csv(os.path.join(MODELOS, "tabla_5_13_proveedores_cri.csv"),
                    index=False)
    print("\n=== Tabla 5.13 ===")
    for _, r in salida13.iterrows():
        print(f"| {r.seudonimo} | {r.familia} | {str(r.depto).title()} | "
              f"{esp(r.cri_score,3)} | {int(r.n_contratos)} | "
              f"{int(r.n_entidades)} | {esp(r.pct_single,2)} | "
              f"{esp(r.pct_directa,2)} |")

    # ---- Tabla 5.14: entidades con más triángulos ----------------------
    #: La definición del triángulo y la exclusión de banca y seguros vienen de
    #: `tfm.grafo.triangulos`: aquí estaban repetidas y sin la condición de que
    #: la unión temporal y el socio sean actores distintos.
    triangulos.preparar(con, PQ)

    t14 = con.execute("""
        SELECT s.seudonimo, e.nombre, e.orden_entidad, e.sector,
               e.departamento_norm AS depto,
               count(*) AS triangulos,
               count(DISTINCT t.ut) AS uts,
               count(DISTINCT t.socio) AS socios
        FROM tri t
        LEFT JOIN ent e ON e.id_nodo = t.entidad
        JOIN s ON s.id_nodo = t.entidad
        GROUP BY 1, 2, 3, 4, 5 ORDER BY 6 DESC LIMIT 8
    """).df()
    #: Para una entidad pública el orden administrativo y el sector describen
    #: mejor de qué se trata que cualquier familia deducida del nombre.
    t14["familia"] = [
        f"{str(o).title()} · {str(sec).title()}" if pd.notna(o) and pd.notna(sec)
        else familia(n)
        for o, sec, n in zip(t14["orden_entidad"], t14["sector"], t14["nombre"])]
    citados += list(zip(t14["seudonimo"], t14["nombre"].fillna("(sin nombre)")))
    salida14 = t14[["seudonimo", "familia", "depto", "triangulos", "uts", "socios"]]
    salida14.to_csv(os.path.join(MODELOS, "tabla_5_14_entidades_triangulos.csv"),
                    index=False)
    print("\n=== Tabla 5.14 ===")
    for _, r in salida14.iterrows():
        print(f"| {r.seudonimo} | {r.familia} | {str(r.depto).title()} | "
              f"{int(r.triangulos)} | {int(r.uts)} | {int(r.socios)} |")

    # ---- Tabla 5.15: pares con más coincidencia ------------------------
    t15 = con.execute("""
        SELECT sa.seudonimo AS a, sb.seudonimo AS b,
               na.nombre AS nombre_a, nb.nombre AS nombre_b, x.peso
        FROM (--: co_presentacion y co_oferta describen el mismo par desde dos
              --: fuentes; se conserva el mayor de los dos pesos y no ambos.
              SELECT origen, destino, max(peso) AS peso FROM v
              WHERE tipo_vinculo IN ('co_presentacion','co_oferta')
              GROUP BY origen, destino
              ORDER BY peso DESC LIMIT 40) x
        JOIN s sa ON sa.id_nodo = x.origen
        JOIN s sb ON sb.id_nodo = x.destino
        LEFT JOIN nodo na ON na.id_nodo = x.origen
        LEFT JOIN nodo nb ON nb.id_nodo = x.destino
        ORDER BY x.peso DESC LIMIT 6
    """).df()
    tipos = con.execute("SELECT id_nodo, tipo_nodo FROM s").df().set_index(
        "id_nodo")["tipo_nodo"].to_dict()
    seu2tipo = con.execute("SELECT seudonimo, tipo_nodo FROM s").df().set_index(
        "seudonimo")["tipo_nodo"].to_dict()
    t15["familia_a"] = [familia(n, seu2tipo.get(a))
                        for n, a in zip(t15["nombre_a"], t15["a"])]
    t15["familia_b"] = [familia(n, seu2tipo.get(b))
                        for n, b in zip(t15["nombre_b"], t15["b"])]
    citados += list(zip(t15["a"], t15["nombre_a"].fillna("(sin nombre)")))
    citados += list(zip(t15["b"], t15["nombre_b"].fillna("(sin nombre)")))
    salida15 = t15[["a", "familia_a", "b", "familia_b", "peso"]]
    salida15.to_csv(os.path.join(MODELOS, "tabla_5_15_pares_coincidencia.csv"),
                    index=False)
    print("\n=== Tabla 5.15 ===")
    for _, r in salida15.iterrows():
        print(f"| {r.a} | {r.familia_a} | {r.b} | {r.familia_b} | {esp(r.peso)} |")

    # ---- Clave restringida ---------------------------------------------
    os.makedirs(SALIDAS, exist_ok=True)
    clave = pd.DataFrame(sorted(set(citados)),
                         columns=["seudonimo", "nombre"])
    f = os.path.join(SALIDAS, "clave_seudonimos_RESTRINGIDO.csv")
    clave.to_csv(f, index=False)
    print(f"\nclave restringida: {len(clave)} actores citados -> {f}")
    print("contiene datos personales: no se versiona ni se distribuye")


if __name__ == "__main__":
    main()
