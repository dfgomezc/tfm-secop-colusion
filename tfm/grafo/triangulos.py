#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""El triángulo de riesgo entidad–UT–socio, definido una sola vez.

Un triángulo de riesgo se cierra cuando una entidad adjudica a una unión
temporal, un socio integra esa unión y **ese mismo socio recibe además una
adjudicación directa de la misma entidad**. Es la figura que sostiene el
apartado 5.3.7 y la bandera roja estructural del 5.2.5.

## Por qué existe este módulo

El triángulo lo consultan cuatro sitios —el brazo clásico, el solapamiento, las
figuras de red y el modelo dimensional—, y todos lo miden sobre las mismas
cifras publicadas. Una definición replicada en cuatro consultas admite que una
se ajuste y las otras no; teniéndola en un único módulo, ajustarla es un cambio
en un solo sitio y los cuatro lo heredan.

La condición que más pesa en el recuento es que la unión temporal y el socio
sean **actores distintos**. El vínculo `integra` del universo contiene 1.021
autolazos, en los que una unión temporal figura como integrante de sí misma;
admitirlos haría que cada uno cerrase un ciclo de dos vértices, que es una
adjudicación contada dos veces y no el acceso por doble vía que el patrón
describe. Con la condición, el universo tiene **1.978** triángulos sobre 539
entidades, 1.626 uniones temporales y 1.239 socios; sin ella, la consulta
devuelve 4.448 ternas.

## Qué expone

    preparar(con)     crea las vistas `exc`, `v`, `adj`, `inte` y `tri`
    actores(con)      el conjunto de id_nodo implicados en algún triángulo
    triangulos(con)   las ternas (entidad, ut, socio)
    recuento(con)     totales, para las cifras del documento

Las vistas quedan disponibles para quien las necesite: el brazo clásico levanta
sobre `v` y `exc` sus otras tres banderas.
"""

#: Exclusión de banca y seguros. Su presencia en un contrato responde a la
#: constitución de garantías y no a una relación de competencia, de modo que
#: incluirlas conecta entre sí a proveedores que no concurren a los mismos
#: procesos.
CIIU_FINANCIERO = ("64", "65", "66")
PATRON_FINANCIERO = ("SEGURO|ASEGURADOR|FIDUCIARI|BANCO|BANCARI|"
                     "CORREDOR DE SEGURO|CAPITALIZADORA")


def sql(pq):
    """El SQL que deja creadas las vistas, sobre el modelo dimensional `pq`."""
    fin = " OR ".join(f"substr(ciiu1, 1, 2) = '{d}'" for d in CIIU_FINANCIERO)
    return f"""
        CREATE OR REPLACE TEMP VIEW exc AS
          SELECT id_nodo FROM '{pq}/nodo_2025/*.parquet'
          WHERE (ciiu1 IS NOT NULL AND ({fin}))
             OR regexp_matches(upper(nombre), '{PATRON_FINANCIERO}');

        CREATE OR REPLACE TEMP VIEW v AS
          SELECT * FROM '{pq}/vinculo_2025/*.parquet';

        CREATE OR REPLACE TEMP VIEW adj AS
          SELECT origen AS entidad, destino AS proveedor FROM v
          WHERE tipo_vinculo = 'adjudica'
            AND destino NOT IN (SELECT id_nodo FROM exc);

        CREATE OR REPLACE TEMP VIEW inte AS
          SELECT origen AS ut, destino AS socio FROM v
          WHERE tipo_vinculo = 'integra'
            AND destino NOT IN (SELECT id_nodo FROM exc)
            AND origen  NOT IN (SELECT id_nodo FROM exc)
            -- La unión temporal y el socio han de ser actores distintos. Sin
            -- esta condición, los 1.021 autolazos del vínculo `integra`
            -- producen 2.470 «triángulos» de dos actores.
            AND origen <> destino;

        CREATE OR REPLACE TEMP VIEW tri AS
          SELECT a.entidad, a.proveedor AS ut, i.socio
          FROM adj a
          JOIN inte i ON i.ut = a.proveedor
          JOIN adj d ON d.entidad = a.entidad AND d.proveedor = i.socio;
    """


def preparar(con, pq):
    con.execute(sql(pq))


def actores(con, pq=None):
    """Las uniones temporales y los socios implicados en algún triángulo."""
    if pq is not None:
        preparar(con, pq)
    return {r[0] for r in con.execute(
        "SELECT ut FROM tri UNION SELECT socio FROM tri").fetchall()}


def triangulos(con, pq=None):
    """Las ternas (entidad, ut, socio), ordenadas de forma reproducible.

    El `ORDER BY` no es cosmético: DuckDB paraleliza y sin orden explícito
    devuelve las filas en un orden distinto en cada ejecución, que es lo que
    ordena después las figuras.
    """
    if pq is not None:
        preparar(con, pq)
    return con.execute("SELECT entidad, ut, socio FROM tri "
                       "ORDER BY 1, 2, 3").df()


def recuento(con, pq=None):
    """Totales del universo, tal como los publica el documento."""
    if pq is not None:
        preparar(con, pq)
    t, e, u, s = con.execute("""
        SELECT count(*), count(DISTINCT entidad),
               count(DISTINCT ut), count(DISTINCT socio) FROM tri
    """).fetchone()
    return {"triangulos": t, "entidades": e, "uniones_temporales": u,
            "socios": s}
