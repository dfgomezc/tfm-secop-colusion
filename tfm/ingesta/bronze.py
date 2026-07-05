#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ingesta JSONL (SECOP II / Socrata) -> Parquet tipado (capa BRONZE).

Que hace:
  * Lee cada carpeta de .jsonl por lotes de ficheros (memoria acotada).
  * Se queda solo con las columnas que el analisis necesita (ver COLUMNAS).
  * Normaliza nulos de SECOP ('No Definido', 'No Provisto', 'No Aplica', ...).
  * Castea fechas, decimales y enteros.
  * Escribe Parquet ZSTD en bronze/<dataset>/part_XXXX.parquet.

Uso:
  python -m tfm.ingesta.bronze --muestra 4              # prueba rapida: 4 chunks por dataset
  python -m tfm.ingesta.bronze                          # volumen completo
  python -m tfm.ingesta.bronze --datasets procesos contratos
  python -m tfm.ingesta.bronze --ram 12GB --hilos 8     # ajustar a la maquina
"""

import argparse
import glob
import os
import sys
import tempfile
import time

import duckdb

from tfm import rutas

# --------------------------------------------------------------------------
# Rutas por defecto (se resuelven relativas a la carpeta padre de este script)
# --------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
#: Por defecto se leen los JSONL dispuestos desde el .zip. Se mantiene
#: `--raiz` para poder apuntar a un volcado en otro disco.
#: De dónde se leen los JSONL y a dónde va la capa bronze. Son dos sitios
#: distintos desde que la frontera del proyecto está en los parquet: `INPUT/`
#: guarda el origen y `OUTPUTS/bronze/` la conversión.
RAIZ_DEF = rutas.entrada()
SALIDA_DEF = rutas.bronze()

# --------------------------------------------------------------------------
# Helpers SQL
# --------------------------------------------------------------------------
# Valores centinela que SECOP usa en lugar de NULL (comparados en mayusculas).
NULOS = ("NO DEFINIDO", "NO PROVISTO", "NO APLICA", "NO ADJUDICADO",
         "SIN DESCRIPCION", "SIN DESCRIPCIÓN", "NO INFORMADO", "N/A", "NA", "-")
_LISTA_NULOS = ", ".join(f"'{n}'" for n in NULOS)


def txt(col, alias=None):
    """VARCHAR limpio: trim + centinelas de SECOP -> NULL."""
    e = f"trim(CAST({col} AS VARCHAR))"
    return (f"CASE WHEN {e} = '' OR upper({e}) IN ({_LISTA_NULOS}) "
            f"THEN NULL ELSE {e} END AS {alias or col}")


def fecha(col, alias=None):
    return f"TRY_CAST(substr(CAST({col} AS VARCHAR), 1, 10) AS DATE) AS {alias or col}"


def fecha8(col, alias=None):
    """Fechas del RUES en formato YYYYMMDD. 99991231 = 'sin vencimiento'."""
    c = f"nullif(trim(CAST({col} AS VARCHAR)), '')"
    return (f"CASE WHEN {c} ~ '^[0-9]{{8}}$' AND {c} <> '99991231' "
            f"THEN TRY_STRPTIME({c}, '%Y%m%d')::DATE END AS {alias or col}")


def num(col, alias=None):
    return (f"TRY_CAST(replace(replace(CAST({col} AS VARCHAR), ',', ''), '$', '') "
            f"AS DECIMAL(24,2)) AS {alias or col}")


def ent(col, alias=None):
    return f"TRY_CAST(CAST({col} AS VARCHAR) AS BIGINT) AS {alias or col}"


def si_no(col, alias=None):
    """'Si'/'No'/'SI'/'NO'/'Verdadero' -> BOOLEAN."""
    c = f"upper(trim(CAST({col} AS VARCHAR)))"
    return (f"CASE WHEN {c} IN ('SI','S','VERDADERO','TRUE','1') THEN TRUE "
            f"WHEN {c} IN ('NO','N','FALSO','FALSE','0') THEN FALSE END AS {alias or col}")


def canon(col, alias):
    """Nombre social canonico: mayusculas, sin tildes, sin sufijos societarios,
    sin puntuacion ni espacios repetidos. Replica la normalizacion del cap. 5.1.1."""
    e = f"upper(strip_accents(CAST({col} AS VARCHAR)))"
    e = f"regexp_replace({e}, '[^A-Z0-9 ]', ' ', 'g')"
    e = (f"regexp_replace({e}, "
         r"'\s+(S A S|SAS|S A|SA|LTDA|LTD|E U|EU|S C A|SCA|S EN C|CIA|Y CIA|"
         r"E S P|ESP|E I C E|EICE|E S E|ESE|S A E S P|EMPRESA UNIPERSONAL|"
         r"SUCURSAL COLOMBIA|EN LIQUIDACION)\s*$'"
         ", '', 'g')")
    e = f"trim(regexp_replace({e}, '\\s+', ' ', 'g'))"
    return f"nullif({e}, '') AS {alias}"


# --------------------------------------------------------------------------
# Definicion de datasets: columnas de origen (tipo lectura) y SELECT de salida
# --------------------------------------------------------------------------
# 'lectura'  -> dict {campo_json: tipo_duckdb} que se pasa a read_json(columns=...)
#               Solo se parsean estas claves; el resto del JSON se ignora.
# 'select'   -> lista de expresiones SQL sobre esas columnas.
# 'dedup'    -> lista de columnas para DISTINCT ON (o None).
# 'porlote'  -> filas objetivo por lote (controla la RAM).

DATASETS = {

    # ----------------------------------------------------------------- #
    # 1. PROCESOS DE CONTRATACION  (fuente principal del capitulo 5 TFM) #
    # ----------------------------------------------------------------- #
    "procesos": dict(
        carpeta="procesos_contratacion_s2",
        porlote=300_000,
        dedup=["id_del_proceso"],
        lectura={
            "id_del_proceso": "VARCHAR", "referencia_del_proceso": "VARCHAR",
            "id_del_portafolio": "VARCHAR", "nombre_del_procedimiento": "VARCHAR",
            "descripci_n_del_procedimiento": "VARCHAR",
            "entidad": "VARCHAR", "nit_entidad": "VARCHAR", "codigo_entidad": "VARCHAR",
            "departamento_entidad": "VARCHAR", "ciudad_entidad": "VARCHAR",
            "ordenentidad": "VARCHAR",
            "fase": "VARCHAR", "estado_del_procedimiento": "VARCHAR",
            "estado_resumen": "VARCHAR", "estado_de_apertura_del_proceso": "VARCHAR",
            "fecha_de_publicacion_del": "VARCHAR", "fecha_de_ultima_publicaci": "VARCHAR",
            "fecha_de_recepcion_de": "VARCHAR", "fecha_adjudicacion": "VARCHAR",
            "precio_base": "VARCHAR", "valor_total_adjudicacion": "VARCHAR",
            "modalidad_de_contratacion": "VARCHAR", "justificaci_n_modalidad_de": "VARCHAR",
            "tipo_de_contrato": "VARCHAR", "subtipo_de_contrato": "VARCHAR",
            "duracion": "VARCHAR", "unidad_de_duracion": "VARCHAR",
            "proveedores_invitados": "VARCHAR", "proveedores_con_invitacion": "VARCHAR",
            "visualizaciones_del": "VARCHAR", "proveedores_que_manifestaron": "VARCHAR",
            "respuestas_al_procedimiento": "VARCHAR", "respuestas_externas": "VARCHAR",
            "conteo_de_respuestas_a_ofertas": "VARCHAR", "proveedores_unicos_con": "VARCHAR",
            "numero_de_lotes": "VARCHAR",
            "adjudicado": "VARCHAR", "id_adjudicacion": "VARCHAR",
            "codigoproveedor": "VARCHAR", "nombre_del_proveedor": "VARCHAR",
            "nit_del_proveedor_adjudicado": "VARCHAR",
            "departamento_proveedor": "VARCHAR", "ciudad_proveedor": "VARCHAR",
            "codigo_principal_de_categoria": "VARCHAR",
            "urlproceso": "JSON",
        },
        select=[
            txt("id_del_proceso"), txt("referencia_del_proceso"),
            txt("id_del_portafolio", "id_portafolio"),
            txt("nombre_del_procedimiento"), txt("descripci_n_del_procedimiento", "descripcion_procedimiento"),
            txt("entidad", "nombre_entidad"), txt("nit_entidad"), txt("codigo_entidad"),
            txt("departamento_entidad"), txt("ciudad_entidad"), txt("ordenentidad", "orden_entidad"),
            txt("fase"), txt("estado_del_procedimiento"), txt("estado_resumen"),
            txt("estado_de_apertura_del_proceso"),
            fecha("fecha_de_publicacion_del", "fecha_publicacion"),
            fecha("fecha_de_ultima_publicaci", "fecha_ultima_publicacion"),
            fecha("fecha_de_recepcion_de", "fecha_cierre_recepcion"),
            fecha("fecha_adjudicacion"),
            num("precio_base"), num("valor_total_adjudicacion"),
            txt("modalidad_de_contratacion"), txt("justificaci_n_modalidad_de", "justificacion_modalidad"),
            txt("tipo_de_contrato"), txt("subtipo_de_contrato"),
            ent("duracion"), txt("unidad_de_duracion"),
            ent("proveedores_invitados"), ent("proveedores_con_invitacion"),
            ent("visualizaciones_del", "visualizaciones"),
            ent("proveedores_que_manifestaron"), ent("respuestas_al_procedimiento"),
            ent("respuestas_externas"), ent("conteo_de_respuestas_a_ofertas"),
            ent("proveedores_unicos_con", "proveedores_unicos_respuesta"),
            ent("numero_de_lotes"),
            si_no("adjudicado"), txt("id_adjudicacion"),
            txt("codigoproveedor", "codigo_proveedor"),
            txt("nombre_del_proveedor", "nombre_proveedor"),
            canon("nombre_del_proveedor", "nombre_proveedor_canonico"),
            txt("nit_del_proveedor_adjudicado", "nit_proveedor"),
            txt("departamento_proveedor"), txt("ciudad_proveedor"),
            txt("codigo_principal_de_categoria", "codigo_categoria"),
            "json_extract_string(urlproceso, '$.url') AS url_proceso",
        ],
    ),

    # ----------------------------------------------------------------- #
    # 2. CONTRATOS ELECTRONICOS  (grano contrato, hecho principal PowerBI)#
    # ----------------------------------------------------------------- #
    "contratos": dict(
        carpeta="contratos_electronicos_s2",
        porlote=200_000,
        dedup=["id_contrato"],
        lectura={
            "id_contrato": "VARCHAR", "referencia_del_contrato": "VARCHAR",
            "proceso_de_compra": "VARCHAR", "estado_contrato": "VARCHAR",
            "nombre_entidad": "VARCHAR", "nit_entidad": "VARCHAR", "codigo_entidad": "VARCHAR",
            "departamento": "VARCHAR", "ciudad": "VARCHAR", "orden": "VARCHAR",
            "sector": "VARCHAR", "rama": "VARCHAR", "entidad_centralizada": "VARCHAR",
            "codigo_de_categoria_principal": "VARCHAR",
            "tipo_de_contrato": "VARCHAR", "modalidad_de_contratacion": "VARCHAR",
            "justificacion_modalidad_de": "VARCHAR",
            "fecha_de_firma": "VARCHAR", "fecha_de_inicio_del_contrato": "VARCHAR",
            "fecha_de_fin_del_contrato": "VARCHAR", "ultima_actualizacion": "VARCHAR",
            "tipodocproveedor": "VARCHAR", "documento_proveedor": "VARCHAR",
            "codigo_proveedor": "VARCHAR", "proveedor_adjudicado": "VARCHAR",
            "es_grupo": "VARCHAR", "es_pyme": "VARCHAR",
            "origen_de_los_recursos": "VARCHAR", "destino_gasto": "VARCHAR",
            "valor_del_contrato": "VARCHAR", "valor_facturado": "VARCHAR",
            "valor_pagado": "VARCHAR", "valor_pendiente_de_pago": "VARCHAR",
            "valor_pendiente_de_ejecucion": "VARCHAR",
            "dias_adicionados": "VARCHAR", "liquidaci_n": "VARCHAR",
            "espostconflicto": "VARCHAR", "objeto_del_contrato": "VARCHAR",
            "nombre_representante_legal": "VARCHAR",
            "identificaci_n_representante_legal": "VARCHAR",
            "nombre_supervisor": "VARCHAR",
            "n_mero_de_documento_supervisor": "VARCHAR",
            "urlproceso": "JSON",
        },
        select=[
            txt("id_contrato"), txt("referencia_del_contrato"),
            txt("proceso_de_compra"), txt("estado_contrato"),
            txt("nombre_entidad"), txt("nit_entidad"), txt("codigo_entidad"),
            txt("departamento", "departamento_entidad"), txt("ciudad", "ciudad_entidad"),
            txt("orden", "orden_entidad"), txt("sector"), txt("rama"),
            txt("entidad_centralizada"),
            txt("codigo_de_categoria_principal", "codigo_categoria"),
            txt("tipo_de_contrato"), txt("modalidad_de_contratacion"),
            txt("justificacion_modalidad_de", "justificacion_modalidad"),
            fecha("fecha_de_firma", "fecha_firma"),
            fecha("fecha_de_inicio_del_contrato", "fecha_inicio"),
            fecha("fecha_de_fin_del_contrato", "fecha_fin"),
            fecha("ultima_actualizacion"),
            txt("tipodocproveedor", "tipo_doc_proveedor"),
            txt("documento_proveedor", "nit_proveedor"),
            txt("codigo_proveedor"), txt("proveedor_adjudicado", "nombre_proveedor"),
            canon("proveedor_adjudicado", "nombre_proveedor_canonico"),
            si_no("es_grupo"), si_no("es_pyme"),
            txt("origen_de_los_recursos"), txt("destino_gasto"),
            num("valor_del_contrato"), num("valor_facturado"), num("valor_pagado"),
            num("valor_pendiente_de_pago"), num("valor_pendiente_de_ejecucion"),
            ent("dias_adicionados"), si_no("liquidaci_n", "liquidacion"),
            si_no("espostconflicto", "es_postconflicto"),
            txt("objeto_del_contrato"),
            txt("nombre_representante_legal", "nombre_rep_legal"),
            txt("identificaci_n_representante_legal", "doc_rep_legal"),
            txt("nombre_supervisor"),
            txt("n_mero_de_documento_supervisor", "doc_supervisor"),
            "json_extract_string(urlproceso, '$.url') AS url_proceso",
        ],
    ),

    # ----------------------------------------------------------------- #
    # 3. PROPONENTES POR PROCESO  (numero real de oferentes / co-licitacion)
    # ----------------------------------------------------------------- #
    "proponentes": dict(
        carpeta="proponentes_por_proceso_s2",
        porlote=500_000,
        dedup=["id_procedimiento", "codigo_proveedor"],
        lectura={
            "id_procedimiento": "VARCHAR", "fecha_publicaci_n": "VARCHAR",
            "nombre_procedimiento": "VARCHAR",
            "nit_entidad": "VARCHAR", "codigo_entidad": "VARCHAR",
            "entidad_compradora": "VARCHAR",
            "proveedor": "VARCHAR", "nit_proveedor": "VARCHAR",
            "codigo_proveedor": "VARCHAR",
        },
        select=[
            txt("id_procedimiento"), fecha("fecha_publicaci_n", "fecha_publicacion"),
            txt("nombre_procedimiento"),
            txt("nit_entidad"), txt("codigo_entidad"), txt("entidad_compradora", "nombre_entidad"),
            txt("proveedor", "nombre_proveedor"),
            canon("proveedor", "nombre_proveedor_canonico"),
            txt("nit_proveedor"), txt("codigo_proveedor"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 4. OFERTAS POR PROCESO  (valor de cada oferta) - MUY DUPLICADO      #
    # ----------------------------------------------------------------- #
    "ofertas": dict(
        carpeta="ofertas_por_proceso_s2",
        porlote=400_000,
        # Redescarga 2026-08-04 con paginacion ordenada: .json (array pretty-
        # printed) en vez de .jsonl, y sin duplicados (antes 22x, ahora 1x).
        extension="json",
        formato="auto",
        dedup=["id_del_proceso_de_compra", "identificador_de_la_oferta", "c_digo_proveedor"],
        lectura={
            "id_del_proceso_de_compra": "VARCHAR", "identificador_de_la_oferta": "VARCHAR",
            "referencia_de_la_oferta": "VARCHAR", "referencia_del_proceso": "VARCHAR",
            "fecha_de_registro": "VARCHAR",
            "valor_de_la_oferta": "VARCHAR", "moneda": "VARCHAR",
            "entidad_compradora": "VARCHAR", "nit_entidad_compradora": "VARCHAR",
            "c_digo_entidad": "VARCHAR",
            "modalidad": "VARCHAR", "invitacion_directa": "VARCHAR",
            "nombre_proveedor": "VARCHAR", "nit_del_proveedor": "VARCHAR",
            "c_digo_proveedor": "VARCHAR",
        },
        select=[
            txt("id_del_proceso_de_compra", "id_del_proceso"),
            txt("identificador_de_la_oferta", "id_oferta"),
            txt("referencia_de_la_oferta"), txt("referencia_del_proceso"),
            fecha("fecha_de_registro", "fecha_registro"),
            num("valor_de_la_oferta", "valor_oferta"), txt("moneda"),
            txt("entidad_compradora", "nombre_entidad"),
            txt("nit_entidad_compradora", "nit_entidad"),
            txt("c_digo_entidad", "codigo_entidad"),
            txt("modalidad", "modalidad_de_contratacion"),
            si_no("invitacion_directa"),
            txt("nombre_proveedor"), canon("nombre_proveedor", "nombre_proveedor_canonico"),
            txt("nit_del_proveedor", "nit_proveedor"),
            txt("c_digo_proveedor", "codigo_proveedor"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 5. GRUPOS DE PROVEEDORES (UT / consorcios y sus socios)             #
    # ----------------------------------------------------------------- #
    "grupos": dict(
        carpeta="grupos_proveedores_s2",
        porlote=400_000,
        dedup=["codigo_grupo", "codigo_participante"],
        lectura={
            "codigo_grupo": "VARCHAR", "nombre_grupo": "VARCHAR", "nit_grupo": "VARCHAR",
            "esta_activo": "VARCHAR", "fecha_creaci_n_grupo": "VARCHAR",
            "tipo_empresa_grupo": "VARCHAR", "departamento_grupo": "VARCHAR",
            "minucipio": "VARCHAR", "ubicaci_n_grupo": "VARCHAR", "es_mipyme": "VARCHAR",
            "correo_electronico_grupo": "VARCHAR", "numero_tel_fono_grupo": "VARCHAR",
            "nombre_representante_legal_grupo": "VARCHAR",
            "numero_doc_representante_legal_grupo": "VARCHAR",
            "codigo_participante": "VARCHAR", "nombre_participante": "VARCHAR",
            "nit_participante": "VARCHAR", "tipo_empresa_participante": "VARCHAR",
            "ubicaci_n_participante": "VARCHAR", "fecha_creaci_n_participante": "VARCHAR",
            "participacion": "VARCHAR", "es_lider_del_grupo": "VARCHAR",
        },
        select=[
            txt("codigo_grupo"), txt("nombre_grupo"),
            canon("nombre_grupo", "nombre_grupo_canonico"),
            txt("nit_grupo"), si_no("esta_activo"),
            fecha("fecha_creaci_n_grupo", "fecha_creacion_grupo"),
            txt("tipo_empresa_grupo"), txt("departamento_grupo"),
            txt("minucipio", "municipio_grupo"), txt("ubicaci_n_grupo", "ubicacion_grupo"),
            si_no("es_mipyme", "es_mipyme_grupo"),
            txt("correo_electronico_grupo", "correo_grupo"),
            txt("numero_tel_fono_grupo", "telefono_grupo"),
            txt("nombre_representante_legal_grupo", "nombre_rep_legal_grupo"),
            txt("numero_doc_representante_legal_grupo", "doc_rep_legal_grupo"),
            txt("codigo_participante"), txt("nombre_participante"),
            canon("nombre_participante", "nombre_participante_canonico"),
            txt("nit_participante"), txt("tipo_empresa_participante"),
            txt("ubicaci_n_participante", "ubicacion_participante"),
            fecha("fecha_creaci_n_participante", "fecha_creacion_participante"),
            num("participacion", "pct_participacion"),
            si_no("es_lider_del_grupo", "es_lider"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 6. PROVEEDORES REGISTRADOS (dimension proveedor)                    #
    # ----------------------------------------------------------------- #
    "proveedores": dict(
        carpeta="proveedores_registrados_s2",
        porlote=500_000,
        dedup=["codigo"],
        lectura={
            "codigo": "VARCHAR", "nombre": "VARCHAR", "nit": "VARCHAR",
            "es_entidad": "VARCHAR", "es_grupo": "VARCHAR", "esta_activa": "VARCHAR",
            "fecha_creacion": "VARCHAR",
            "codigo_categoria_principal": "VARCHAR", "descripcion_categoria_principal": "VARCHAR",
            "telefono": "VARCHAR", "fax": "VARCHAR", "correo": "VARCHAR",
            "direccion": "VARCHAR", "pais": "VARCHAR", "departamento": "VARCHAR",
            "municipio": "VARCHAR", "sitio_web": "VARCHAR", "tipo_empresa": "VARCHAR",
            "nombre_representante_legal": "VARCHAR",
            "tipo_doc_representante_legal": "VARCHAR",
            "n_mero_doc_representante_legal": "VARCHAR",
            "telefono_representante_legal": "VARCHAR",
            "correo_representante_legal": "VARCHAR",
            "espyme": "VARCHAR", "ubicacion": "VARCHAR",
        },
        select=[
            txt("codigo", "codigo_proveedor"), txt("nombre", "nombre_proveedor"),
            canon("nombre", "nombre_proveedor_canonico"),
            txt("nit", "nit_proveedor"),
            si_no("es_entidad"), si_no("es_grupo"), si_no("esta_activa"),
            fecha("fecha_creacion"),
            txt("codigo_categoria_principal", "codigo_categoria"),
            txt("descripcion_categoria_principal", "descripcion_categoria"),
            txt("telefono"), txt("fax"), txt("correo"), txt("direccion"),
            txt("pais"), txt("departamento"), txt("municipio"),
            txt("sitio_web"), txt("tipo_empresa"),
            txt("nombre_representante_legal", "nombre_rep_legal"),
            txt("tipo_doc_representante_legal", "tipo_doc_rep_legal"),
            txt("n_mero_doc_representante_legal", "doc_rep_legal"),
            txt("telefono_representante_legal", "telefono_rep_legal"),
            txt("correo_representante_legal", "correo_rep_legal"),
            si_no("espyme", "es_pyme"), txt("ubicacion"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 7. DATOS DE CONTACTO (dimension entidad + señales de contacto)      #
    # ----------------------------------------------------------------- #
    "contacto": dict(
        carpeta="datos_de_contacto_s2",
        porlote=500_000,
        dedup=["codigo_entidad"],
        lectura={
            "codigo_entidad": "VARCHAR", "nombre_entidad": "VARCHAR", "nit_entidad": "VARCHAR",
            "es_entidad": "VARCHAR", "es_proveedor": "VARCHAR", "es_grupo": "VARCHAR",
            "esta_activa": "VARCHAR", "feacha_de_creacion": "VARCHAR",
            "codigo_categoria_principal": "VARCHAR", "descripci_n_categoria_principal": "VARCHAR",
            "numero_fax": "VARCHAR", "correo_electronico": "VARCHAR",
            "pais": "VARCHAR", "departamento": "VARCHAR", "ciudad": "VARCHAR",
            "website": "VARCHAR", "tipo_entidad": "VARCHAR",
            "nombre_representante_legal": "VARCHAR",
            "tipo_documento_representante_legal": "VARCHAR",
            "n_mero_documento_representante_legal": "VARCHAR",
            "correo_representante_legal": "VARCHAR",
            "es_pyme": "VARCHAR", "c_digo_ubicaci_n": "VARCHAR",
        },
        select=[
            txt("codigo_entidad"), txt("nombre_entidad"), txt("nit_entidad"),
            si_no("es_entidad"), si_no("es_proveedor"), si_no("es_grupo"),
            si_no("esta_activa"), fecha("feacha_de_creacion", "fecha_creacion"),
            txt("codigo_categoria_principal", "codigo_categoria"),
            txt("descripci_n_categoria_principal", "descripcion_categoria"),
            txt("numero_fax", "fax"), txt("correo_electronico", "correo"),
            txt("pais"), txt("departamento"), txt("ciudad"),
            txt("website"), txt("tipo_entidad"),
            txt("nombre_representante_legal", "nombre_rep_legal"),
            txt("tipo_documento_representante_legal", "tipo_doc_rep_legal"),
            txt("n_mero_documento_representante_legal", "doc_rep_legal"),
            txt("correo_representante_legal", "correo_rep_legal"),
            si_no("es_pyme"), txt("c_digo_ubicaci_n", "codigo_ubicacion"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 9. MATRICULAS RUES (atributos de nodo: edad, actividad, ubicacion)  #
    # ----------------------------------------------------------------- #
    "rues": dict(
        carpeta="matriculas_rues",
        porlote=400_000,
        dedup=["codigo_camara", "matricula"],
        lectura={
            "codigo_camara": "VARCHAR", "camara_comercio": "VARCHAR",
            "matricula": "VARCHAR", "inscripcion_proponente": "VARCHAR",
            "razon_social": "VARCHAR", "sigla": "VARCHAR",
            "codigo_clase_identificacion": "VARCHAR", "clase_identificacion": "VARCHAR",
            "numero_identificacion": "VARCHAR", "digito_verificacion": "VARCHAR",
            "nit": "VARCHAR",
            "cod_ciiu_act_econ_pri": "VARCHAR", "cod_ciiu_act_econ_sec": "VARCHAR",
            "ciiu3": "VARCHAR", "ciiu4": "VARCHAR",
            "fecha_matricula": "VARCHAR", "fecha_renovacion": "VARCHAR",
            "ultimo_ano_renovado": "VARCHAR", "fecha_vigencia": "VARCHAR",
            "fecha_cancelacion": "VARCHAR", "fecha_actualizacion": "VARCHAR",
            "codigo_tipo_sociedad": "VARCHAR", "tipo_sociedad": "VARCHAR",
            "codigo_organizacion_juridica": "VARCHAR", "organizacion_juridica": "VARCHAR",
            "codigo_categoria_matricula": "VARCHAR", "categoria_matricula": "VARCHAR",
            "codigo_estado_matricula": "VARCHAR", "estado_matricula": "VARCHAR",
            "representante_legal": "VARCHAR", "clase_identificacion_rl": "VARCHAR",
            "num_identificacion_representante_legal": "VARCHAR",
            "primer_nombre": "VARCHAR", "segundo_nombre": "VARCHAR",
            "primer_apellido": "VARCHAR", "segundo_apellido": "VARCHAR",
        },
        select=[
            txt("codigo_camara"), txt("camara_comercio"),
            txt("matricula"), txt("inscripcion_proponente"),
            txt("razon_social"), canon("razon_social", "razon_social_canonica"),
            txt("sigla"),
            txt("codigo_clase_identificacion"), txt("clase_identificacion"),
            txt("numero_identificacion"), txt("digito_verificacion"), txt("nit"),
            txt("cod_ciiu_act_econ_pri", "ciiu1"),
            txt("cod_ciiu_act_econ_sec", "ciiu2"),
            txt("ciiu3"), txt("ciiu4"),
            fecha8("fecha_matricula"), fecha8("fecha_renovacion"),
            ent("ultimo_ano_renovado"), fecha8("fecha_vigencia"),
            fecha8("fecha_cancelacion"),
            "TRY_STRPTIME(substr(CAST(fecha_actualizacion AS VARCHAR), 1, 19), "
            "'%Y/%m/%d %H:%M:%S') AS fecha_actualizacion",
            txt("codigo_tipo_sociedad"), txt("tipo_sociedad"),
            txt("codigo_organizacion_juridica"), txt("organizacion_juridica"),
            txt("codigo_categoria_matricula"), txt("categoria_matricula"),
            txt("codigo_estado_matricula"), txt("estado_matricula"),
            txt("representante_legal"),
            canon("representante_legal", "representante_legal_canonico"),
            txt("clase_identificacion_rl"),
            txt("num_identificacion_representante_legal", "doc_representante_legal"),
            txt("primer_nombre"), txt("segundo_nombre"),
            txt("primer_apellido"), txt("segundo_apellido"),
        ],
    ),

    # ----------------------------------------------------------------- #
    # 8. MULTAS Y SANCIONES (ground truth parcial / validacion externa)   #
    # ----------------------------------------------------------------- #
    "multas": dict(
        carpeta="multas_sanciones_s2",
        porlote=500_000,
        dedup=None,
        lectura={
            "id_proceso": "VARCHAR", "referencia_proceso": "VARCHAR", "id_contrato": "VARCHAR",
            "codigo_entidad_creadora": "VARCHAR", "nombre_entidad_creadora": "VARCHAR",
            "as_codigo_proveedor_objeto": "VARCHAR", "nombre_proveedor_objeto_de": "VARCHAR",
            "valor": "VARCHAR", "valor_pagado": "VARCHAR", "fecha_evento": "VARCHAR",
            "aplico_garantias": "VARCHAR", "numero_de_acto": "VARCHAR",
            "tipo_de_sancion": "VARCHAR", "descripcion_otro_tipo_de": "VARCHAR",
            "estado": "VARCHAR", "tipo": "VARCHAR", "numero_de_version": "VARCHAR",
        },
        select=[
            txt("id_proceso", "id_del_proceso"), txt("referencia_proceso"), txt("id_contrato"),
            txt("codigo_entidad_creadora", "codigo_entidad"),
            txt("nombre_entidad_creadora", "nombre_entidad"),
            txt("as_codigo_proveedor_objeto", "codigo_proveedor"),
            txt("nombre_proveedor_objeto_de", "nombre_proveedor"),
            canon("nombre_proveedor_objeto_de", "nombre_proveedor_canonico"),
            num("valor", "valor_sancion"), num("valor_pagado"),
            fecha("fecha_evento"), si_no("aplico_garantias"),
            txt("numero_de_acto"), txt("tipo_de_sancion"),
            txt("descripcion_otro_tipo_de", "descripcion_otro_tipo"),
            txt("estado"), txt("tipo"), ent("numero_de_version"),
        ],
    ),
}


# --------------------------------------------------------------------------
def limpiar(ruta):
    """Borra un fichero; si el sistema no lo permite (carpetas protegidas),
    lo deja en 0 bytes. Los lectores ignoran los ficheros vacios."""
    try:
        os.remove(ruta)
    except OSError:
        try:
            open(ruta, "wb").close()
        except OSError:
            pass


def parquet_valido(ruta):
    """True si el fichero existe y termina con el magic 'PAR1' (no quedo a
    medias por una interrupcion)."""
    try:
        if os.path.getsize(ruta) < 8:
            return False
        with open(ruta, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def lotes(lista, n):
    for i in range(0, len(lista), n):
        yield lista[i:i + n]


def procesar(con, nombre, cfg, raiz, salida, muestra, reanudar=False, fin=0):
    origen = os.path.join(raiz, cfg["carpeta"])
    extension = cfg.get("extension", "jsonl")
    formato_json = cfg.get("formato", "newline_delimited")
    ficheros = sorted(glob.glob(os.path.join(origen, f"*.{extension}")))
    if not ficheros:
        print(f"  [!] sin ficheros en {origen}")
        return 0

    # filas por chunk (se asume homogeneo dentro del dataset)
    if formato_json == "newline_delimited":
        with open(ficheros[0], "r", encoding="utf-8", errors="ignore") as fh:
            filas_chunk = sum(1 for _ in fh)
    else:
        # ficheros con JSON "pretty-printed" (array): una linea no es un registro,
        # hay que contar via DuckDB.
        filas_chunk = con.execute(
            f"SELECT count(*) FROM read_json('{ficheros[0]}', "
            f"format='{formato_json}', ignore_errors=true)").fetchone()[0]
    por_lote = max(1, cfg["porlote"] // max(filas_chunk, 1))

    if muestra:
        ficheros = ficheros[:muestra]

    destino = os.path.join(salida, nombre)
    os.makedirs(destino, exist_ok=True)
    if not reanudar:
        for viejo in glob.glob(os.path.join(destino, "*.parquet")):
            limpiar(viejo)

    cols = ", ".join(f"'{k}': '{v}'" for k, v in cfg["lectura"].items())
    proj = ",\n       ".join(cfg["select"])

    total, t0, hechos, pendientes = 0, time.time(), 0, 0
    for i, grupo in enumerate(lotes(ficheros, por_lote)):
        parte = os.path.join(destino, f"part_{i:04d}.parquet")
        if reanudar and parquet_valido(parte):
            total += con.execute(
                f"SELECT count(*) FROM read_parquet('{parte}')").fetchone()[0]
            hechos += 1
            continue
        if fin and time.time() > fin:
            pendientes += 1
            continue

        lst = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in grupo) + "]"
        fuente = (f"read_json({lst}, format='{formato_json}', "
                  f"columns={{{cols}}}, ignore_errors=true, "
                  f"maximum_object_size=33554432)")
        if cfg["dedup"]:
            claves = ", ".join(cfg["dedup"])
            sel = (f"SELECT * FROM (SELECT {proj}, "
                   f"row_number() OVER (PARTITION BY {claves}) AS _rn FROM {fuente}) "
                   f"WHERE _rn = 1")
            sel = f"SELECT * EXCLUDE (_rn) FROM ({sel})"
        else:
            sel = f"SELECT {proj} FROM {fuente}"

        con.execute(f"COPY ({sel}) TO '{parte}' "
                    f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{parte}')").fetchone()[0]
        total += n
        print(f"    lote {i:>4} | {len(grupo):>3} ficheros | {n:>9,} filas | "
              f"{time.time()-t0:6.1f}s", flush=True)

    estado = f" | {pendientes} lotes PENDIENTES" if pendientes else " | COMPLETO"
    if hechos:
        estado += f" | {hechos} lotes reutilizados"
    print(f"  -> {nombre}: {total:,} filas en {destino} "
          f"({time.time()-t0:.1f}s){estado}")
    return total, pendientes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=RAIZ_DEF,
                    help="carpeta con los subdirectorios *_s2")
    ap.add_argument("--salida", default=None,
                    help="carpeta destino; por defecto OUTPUTS/bronze")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--muestra", type=int, default=0,
                    help="usar solo los N primeros chunks de cada dataset (0 = todos)")
    ap.add_argument("--ram", default="6GB")
    ap.add_argument("--hilos", type=int, default=0)
    ap.add_argument("--temp", default=None, help="carpeta de spill de DuckDB")
    ap.add_argument("--reanudar", action="store_true",
                    help="conserva los lotes ya escritos y sigue por donde iba")
    ap.add_argument("--limite-seg", type=int, default=0, dest="limite",
                    help="presupuesto global en segundos (usar con --reanudar)")
    a = ap.parse_args()

    salida = a.salida or SALIDA_DEF
    os.makedirs(salida, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.ram}'")
    con.execute("SET preserve_insertion_order=false")
    if a.hilos:
        con.execute(f"SET threads={a.hilos}")
    # El spill debe ir a disco LOCAL rapido; nunca a la carpeta de salida si
    # esta en una unidad de red o montada (lo vuelve inutilizablemente lento).
    tmp = a.temp or os.path.join(tempfile.gettempdir(), "duckdb_secop")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")

    print(f"BRONZE  origen={a.raiz}\n        destino={salida}\n"
          f"        muestra={a.muestra or 'completo'}  ram={a.ram}\n")
    fin = (time.time() + a.limite) if a.limite else 0
    resumen, falta = {}, 0
    for nombre in a.datasets:
        print(f"[{nombre}]")
        try:
            n, p = procesar(con, nombre, DATASETS[nombre], a.raiz, salida,
                            a.muestra, a.reanudar, fin)
            resumen[nombre] = n
            falta += p
        except Exception as e:
            print(f"  [ERROR] {nombre}: {e}", file=sys.stderr)
            resumen[nombre] = -1
    print("\nRESUMEN")
    for k, v in resumen.items():
        print(f"  {k:<14} {v:>12,}" if v >= 0 else f"  {k:<14}        ERROR")
    if falta:
        print(f"\n[!] quedan {falta} lotes; vuelve a lanzar con --reanudar")
    sys.exit(2 if falta else 0)


if __name__ == "__main__":
    main()
