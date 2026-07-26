#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Llave sustituta opaca para los actores que aparecen en los resultados.

Los identificadores de actor no son publicables: el valor de `id_nodo` es
`DOC:<documento>` o `COD:<código>`, de modo que para las 35.511 personas
naturales del universo el identificador es la cédula y para las personas
jurídicas es el NIT. Difundirlo equivale a difundir el documento de identidad,
con el agravante de que la forma técnica del campo lo disimula.

En su lugar se asigna un correlativo con prefijo por tipo de actor, ordenando
por HMAC-SHA256(id_nodo, sal). El orden no es alfabético ni por puntuación de
riesgo: un seudónimo ordenado por riesgo filtraría información sobre el actor
al que designa.

La sal no se publica. Vive en `config/sal_seudonimo.txt`, excluido del control
de versiones. Sin ella la correspondencia no puede recomputarse aunque se
conozca el algoritmo, que es lo que distingue una seudonimización efectiva de
una cosmética.

Salidas:
    OUTPUTS/gold/nodo_seudonimo/nodo_seudonimo.parquet   correspondencia completa
    salidas/clave_seudonimos_RESTRINGIDO.csv             solo los actores citados

El segundo fichero contiene datos de carácter personal y queda fuera del
control de versiones y de cualquier distribución.

Uso:
    python -m tfm.identidad.seudonimizar
    python -m tfm.identidad.seudonimizar --clave    # regenera la clave restringida
"""

import argparse
import hashlib
import hmac
import os
import secrets

import duckdb

from tfm import rutas

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
PQ = rutas.parquet()
CONFIG = os.path.join(rutas.de("config"))
SAL = os.path.join(CONFIG, "sal_seudonimo.txt")
SALIDAS = os.path.join(rutas.salida())

#: Prefijo y anchura del correlativo por tipo de actor. La anchura se fija de
#: antemano para que el seudónimo no revele el tamaño del grupo al que
#: pertenece el actor.
FORMATO = {
    "entidad": ("E", 4),
    "persona_juridica": ("P", 5),
    "union_temporal": ("UT", 4),
    "persona_natural": ("N", 5),
    "sin_clasificar": ("X", 4),
}


def obtener_sal():
    """Lee la sal o la crea si es la primera ejecución.

    Se genera con `secrets`, no con `random`: el objetivo es que no pueda
    reproducirse desde una semilla conocida.
    """
    os.makedirs(CONFIG, exist_ok=True)
    if os.path.exists(SAL):
        with open(SAL, "rb") as fh:
            return fh.read().strip()
    valor = secrets.token_hex(32).encode()
    with open(SAL, "wb") as fh:
        fh.write(valor)
    print(f"  sal creada en {SAL}")
    print("  no debe versionarse ni distribuirse")
    return valor


def proteger_gitignore():
    """`config/` fuera del repositorio: ahí vive la sal."""
    ruta = os.path.join(RAIZ, ".gitignore")
    lineas = []
    if os.path.exists(ruta):
        lineas = open(ruta, encoding="utf-8").read().splitlines()
    faltan = [p for p in ("config/", "salidas/clave_seudonimos_RESTRINGIDO.csv")
              if p not in lineas]
    if faltan:
        with open(ruta, "a", encoding="utf-8") as fh:
            fh.write("\n# Seudonimización: la sal y la clave no se publican\n")
            fh.write("\n".join(faltan) + "\n")
        print(f"  añadido a .gitignore: {', '.join(faltan)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clave", action="store_true",
                    help="regenera la clave restringida de los actores citados")
    a = ap.parse_args()

    sal = obtener_sal()
    proteger_gitignore()

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    nodos = con.execute(f"""
        SELECT id_nodo, tipo_nodo FROM '{PQ}/nodo_2025/*.parquet'
        UNION ALL
        SELECT DISTINCT origen AS id_nodo, 'entidad' AS tipo_nodo
        FROM '{PQ}/vinculo_2025/*.parquet' WHERE tipo_vinculo = 'adjudica'
    """).df().drop_duplicates(subset="id_nodo")

    #: El correlativo se asigna por el orden que induce el HMAC, no por el del
    #: fichero: dos ejecuciones con la misma sal dan el mismo resultado, y sin
    #: la sal el orden es impredecible.
    nodos["_h"] = [hmac.new(sal, n.encode(), hashlib.sha256).hexdigest()
                   for n in nodos["id_nodo"]]
    nodos = nodos.sort_values(["tipo_nodo", "_h"]).reset_index(drop=True)

    seudos = []
    for tipo, grupo in nodos.groupby("tipo_nodo", sort=False):
        pref, ancho = FORMATO.get(tipo, ("X", 5))
        for i, _ in enumerate(grupo.index, start=1):
            seudos.append(f"{pref}-{i:0{ancho}d}")
    nodos = nodos.sort_values(["tipo_nodo", "_h"])
    nodos["seudonimo"] = seudos

    salida = os.path.join(PQ, "nodo_seudonimo")
    os.makedirs(salida, exist_ok=True)
    tabla = nodos[["id_nodo", "tipo_nodo", "seudonimo"]]
    con.register("t", tabla)
    con.execute(f"COPY t TO '{salida}/nodo_seudonimo.parquet' (FORMAT PARQUET)")
    print(f"\n  {len(tabla):,} actores seudonimizados")
    print(tabla.groupby("tipo_nodo").size().to_string())

    if a.clave:
        #: La clave restringida solo cubre a los actores que el documento
        #: nombra: unas decenas, no 140.000.
        os.makedirs(SALIDAS, exist_ok=True)
        nom = con.execute(f"""
            SELECT n.id_nodo, n.nombre, s.seudonimo, s.tipo_nodo
            FROM '{PQ}/nodo_2025/*.parquet' n
            JOIN t s USING (id_nodo)
            WHERE n.n_contratos_2025 >= 40
               OR n.id_nodo IN (SELECT origen FROM '{PQ}/vinculo_2025/*.parquet'
                                WHERE tipo_vinculo = 'comparte_contacto')
            ORDER BY s.seudonimo
        """).df()
        f = os.path.join(SALIDAS, "clave_seudonimos_RESTRINGIDO.csv")
        nom.to_csv(f, index=False)
        print(f"\n  clave restringida: {len(nom):,} actores -> {f}")
        print("  contiene datos personales: no se versiona ni se distribuye")


if __name__ == "__main__":
    main()
