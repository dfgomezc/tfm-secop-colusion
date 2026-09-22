#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dos comprobaciones de sensibilidad pedidas por la evaluación de contenidos.

Ninguna de las dos cambia un resultado del trabajo. Las dos responden a la
pregunta de si una decisión declarada aguanta que se la mire de cerca, y por eso
viven en `verificacion/` y no en el camino de ejecución: se corren cuando hay
que sostener una decisión, no cada vez que se rehace el análisis.

## Qué mide cada una

**Criterio de exclusión de banca y seguros.** El apartado 5.1.6 excluye a los
actores financieros con dos condiciones unidas por O: el código CIIU de división
64, 65 o 66 del registro mercantil, y un patrón sobre la razón social. La
objeción razonable es que el patrón sea una lista *ad hoc* que mueva el
resultado a conveniencia. La comprobación reconstruye el recuento de triángulos
con el criterio reproducible solo —CIIU del RUES— y lo compara con el publicado.

**Reparto de la atribución entre el actor y su entorno, por decil.** El
apartado 5.4 publica el reparto medio sobre las cien alertas explicadas. La
objeción es que un promedio esconda una dependencia con la puntuación: que el
entorno pese sobre todo en la cola alta, por ejemplo, lo que cambiaría cómo se
lee una alerta según dónde caiga. La comprobación parte las cien alertas en
deciles de puntuación y mide el reparto en cada uno.

## Uso

    python -m tfm.verificacion.sensibilidad --que exclusion
    python -m tfm.verificacion.sensibilidad --que atribucion
    python -m tfm.verificacion.sensibilidad --que ambas

Escribe `tabla_5_x_sensibilidad_exclusion.csv` y
`tabla_5_x_atribucion_decil.csv` en `OUTPUTS/tablas/`.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tfm import rutas                                        # noqa: E402
from tfm.grafo import triangulos as tri                      # noqa: E402



def _clave_actor(df):
    """El nombre de la columna que identifica al actor en un marco de datos.

    Las tablas de explicabilidad del conjunto distribuido traen `seudonimo` y
    no `id_nodo`, porque el identificador real no se publica. Las dos designan
    al mismo actor y sirven igual para agrupar.
    """
    for c in ("id_nodo", "seudonimo"):
        if c in df.columns:
            return c
    raise SystemExit(
        "las tablas de explicabilidad no traen columna de actor; "
        "ejecuta antes `python -m tfm.modelos.explicabilidad --comparar`")

def _tablas(nombre):
    destino = rutas.outputs_de("tablas")
    os.makedirs(destino, exist_ok=True)
    return os.path.join(destino, nombre)


# --- 1. sensibilidad del criterio de exclusion -----------------------------

def _sql_solo_ciiu(pq):
    """El mismo SQL del módulo canónico, sin el patrón de razón social.

    Se reescribe la vista `exc` y se reutilizan las cuatro restantes tal como
    las define `tfm.grafo.triangulos`, de modo que la única diferencia entre
    las dos mediciones sea el criterio de exclusión y no la consulta.
    """
    fin = " OR ".join(f"substr(ciiu1, 1, 2) = '{d}'"
                      for d in tri.CIIU_FINANCIERO)
    completo = tri.sql(pq)
    corte = completo.index("CREATE OR REPLACE TEMP VIEW v AS")
    return f"""
        CREATE OR REPLACE TEMP VIEW exc AS
          SELECT id_nodo FROM '{pq}/nodo_2025/*.parquet'
          WHERE ciiu1 IS NOT NULL AND ({fin});
    """ + completo[corte:]


def exclusion():
    import duckdb

    pq = rutas.outputs_de("gold")
    con = duckdb.connect()

    filas = []
    for etiqueta, consulta in (
            ("Solo CIIU 64/65/66 del RUES", _sql_solo_ciiu(pq)),
            ("CIIU + patrón de razón social (publicado)", tri.sql(pq))):
        con.execute(consulta)
        excluidos = con.execute("SELECT count(*) FROM exc").fetchone()[0]
        r = tri.recuento(con)
        filas.append({
            "Criterio de exclusión": etiqueta,
            "Actores excluidos": excluidos,
            "Triángulos": r["triangulos"],
            "Entidades": r["entidades"],
            "Uniones temporales": r["uniones_temporales"],
            "Socios": r["socios"],
        })

    d = pd.DataFrame(filas)
    a, b = d["Triángulos"]
    d.to_csv(_tablas("tabla_5_x_sensibilidad_exclusion.csv"), index=False)
    print(d.to_string(index=False))
    print(f"\ndiferencia: {b - a:+d} triángulos "
          f"({100 * (b - a) / a:+.2f} %)")
    return d


# --- 2. atribucion al entorno por decil de puntuacion ----------------------

def atribucion(modelos=("mlp", "rf")):
    filas = []
    for modelo in modelos:
        base = rutas.outputs_de("modelos", "explicabilidad", modelo)
        var = pd.read_csv(os.path.join(base, "atribucion_variables.csv"))
        bloq = pd.read_csv(os.path.join(base, "atribucion_bloques.csv"))

        #: El porcentaje se calcula igual que el reparto medio que publica el
        #: apartado 5.4: sobre el valor absoluto de las atribuciones, porque
        #: una aportación negativa también explica.
        var["abs"] = var["atribucion"].abs()
        clave = _clave_actor(var)
        total = var.groupby(clave)["abs"].sum()
        entorno = var[var["bloque"] != "propio"].groupby(clave)["abs"].sum()
        pct = (100 * entorno / total).rename("pct_entorno")

        d = bloq[[clave, "puntuacion"]].merge(pct, on=clave)
        #: `rank` antes de `qcut` porque las puntuaciones empatan en la cola
        #: alta y sin él los deciles quedan desiguales.
        d["decil"] = pd.qcut(d["puntuacion"].rank(method="first"), 10,
                             labels=range(1, 11))
        g = d.groupby("decil", observed=True).agg(
            alertas=("pct_entorno", "size"),
            punt_min=("puntuacion", "min"),
            punt_max=("puntuacion", "max"),
            pct_medio=("pct_entorno", "mean"),
            pct_min=("pct_entorno", "min"),
            pct_max=("pct_entorno", "max"),
            manda_entorno=("pct_entorno", lambda s: int((s > 50).sum())),
        ).reset_index()
        g.insert(0, "modelo", modelo)
        filas.append(g)

        rho = d["puntuacion"].corr(d["pct_entorno"], method="spearman")
        print(f"\n{modelo}: reparto medio al entorno "
              f"{d['pct_entorno'].mean():.2f} % · "
              f"Spearman puntuación~%entorno = {rho:+.4f}")
        print(g.round(2).to_string(index=False))

    d = pd.concat(filas, ignore_index=True)
    d.to_csv(_tablas("tabla_5_x_atribucion_decil.csv"), index=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--que", choices=("exclusion", "atribucion", "ambas"),
                    default="ambas")
    a = ap.parse_args()
    if a.que in ("exclusion", "ambas"):
        exclusion()
    if a.que in ("atribucion", "ambas"):
        atribucion()


if __name__ == "__main__":
    main()
