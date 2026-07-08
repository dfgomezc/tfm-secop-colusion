-- ==========================================================================
-- 02_gold.sql — Modelo dimensional SECOP II para el TFM (capa GOLD)
--
-- TFM: "Analisis comparativo entre tecnicas clasicas de grafos y GNN para la
--       deteccion de colusion en el SECOP II de Colombia"
--       O.L. Correa Ramos y D.F. Gomez Cubillos — UNIR
--
-- Se ejecuta desde 02_gold.py, que define la macro/variable ${BRONZE}.
-- Grano principal: un contrato por fila (fact_contrato).
--
-- Esquema:
--   dim_tiempo, dim_departamento, dim_entidad, dim_proveedor,
--   dim_categoria, dim_modalidad, dim_tipo_contrato
--   fact_contrato   (grano: id_contrato)
--   fact_proceso    (grano: id_del_proceso)      <- base del cap. 5 del TFM
--   fact_proponente (grano: proceso x proveedor) <- oferentes reales
--   fact_oferta     (grano: proceso x oferta x proveedor)
--   fact_sancion    (grano: sancion)
--   bridge_ut_socio (grano: UT x socio)
--   + vistas analiticas de competencia, CRI y aristas de grafo
-- ==========================================================================

-- Normalizador de texto para claves de union (mayusculas, sin tildes/ruido).
CREATE OR REPLACE MACRO norm(s) AS
    nullif(trim(regexp_replace(upper(strip_accents(CAST(s AS VARCHAR))),
                               '\s+', ' ', 'g')), '');

-- Normalizador de NIT: quita puntos, guiones y digito de verificacion suelto.
CREATE OR REPLACE MACRO nitn(s) AS
    nullif(regexp_replace(CAST(s AS VARCHAR), '[^0-9]', '', 'g'), '');


-- ==========================================================================
-- DIMENSIONES
-- ==========================================================================

-- --- dim_tiempo -----------------------------------------------------------
CREATE OR REPLACE TABLE dim_tiempo AS
WITH rango AS (
    SELECT least(
             coalesce((SELECT min(fecha_firma)      FROM stg_contratos), DATE '2015-01-01'),
             coalesce((SELECT min(fecha_publicacion) FROM stg_procesos),  DATE '2015-01-01')
           ) AS d0,
           greatest(
             coalesce((SELECT max(fecha_firma)      FROM stg_contratos), DATE '2026-12-31'),
             coalesce((SELECT max(fecha_publicacion) FROM stg_procesos),  DATE '2026-12-31')
           ) AS d1
)
SELECT
    d                                             AS fecha,
    CAST(strftime(d, '%Y%m%d') AS INTEGER)        AS id_fecha,
    year(d)                                       AS anio,
    quarter(d)                                    AS trimestre,
    'T' || quarter(d)                             AS trimestre_txt,
    month(d)                                      AS mes,
    strftime(d, '%Y-%m')                          AS anio_mes,
    monthname(d)                                  AS nombre_mes,
    day(d)                                        AS dia,
    dayofweek(d)                                  AS dia_semana,
    dayname(d)                                    AS nombre_dia_semana,
    week(d)                                       AS semana_iso,
    dayofweek(d) IN (0, 6)                        AS es_fin_de_semana,
    -- ultimo trimestre del anio: pico de contratacion, relevante para el analisis
    quarter(d) = 4                                AS es_cierre_vigencia
FROM rango, range(
        CAST(date_trunc('year', (SELECT d0 FROM rango)) AS TIMESTAMP),
        CAST(date_trunc('year', (SELECT d1 FROM rango)) + INTERVAL 1 YEAR AS TIMESTAMP),
        INTERVAL 1 DAY) t(dts),
     LATERAL (SELECT CAST(dts AS DATE)) v(d);


-- --- dim_departamento -----------------------------------------------------
CREATE OR REPLACE TABLE dim_departamento AS
WITH crudo AS (
    SELECT departamento_entidad AS d FROM stg_contratos
    UNION ALL SELECT departamento_entidad FROM stg_procesos
    UNION ALL SELECT departamento_proveedor FROM stg_procesos
    UNION ALL SELECT departamento FROM stg_proveedores
    UNION ALL SELECT departamento FROM stg_contacto
),
n AS (SELECT DISTINCT norm(d) AS departamento_norm FROM crudo WHERE d IS NOT NULL)
SELECT
    departamento_norm,
    -- nombre de presentacion homogeneo para los mapas de Power BI
    CASE
      WHEN departamento_norm LIKE '%BOGOT%'        THEN 'Bogotá D.C.'
      WHEN departamento_norm LIKE '%SAN ANDRES%'   THEN 'San Andrés y Providencia'
      WHEN departamento_norm LIKE '%VALLE%'        THEN 'Valle del Cauca'
      WHEN departamento_norm LIKE '%NORTE DE SANT%' THEN 'Norte de Santander'
      ELSE list_aggregate(
             list_transform(string_split(lower(departamento_norm), ' '),
                            w -> upper(substr(w, 1, 1)) || substr(w, 2)),
             'string_agg', ' ')
    END                                            AS departamento,
    departamento_norm IN ('DISTRITO CAPITAL DE BOGOTA', 'BOGOTA D.C.', 'BOGOTA')
                                                   AS es_capital
FROM n;


-- --- dim_entidad ----------------------------------------------------------
-- Clave: codigo_entidad (identidad de plataforma, estable; el NIT se repite
-- entre unidades ejecutoras y por eso no sirve como PK).
CREATE OR REPLACE TABLE dim_entidad AS
WITH desde_contratos AS (
    SELECT codigo_entidad,
           any_value(nombre_entidad)      AS nombre_entidad,
           any_value(nit_entidad)         AS nit_entidad,
           any_value(departamento_entidad) AS departamento,
           any_value(ciudad_entidad)      AS ciudad,
           any_value(orden_entidad)       AS orden,
           any_value(sector)              AS sector,
           any_value(rama)                AS rama,
           any_value(entidad_centralizada) AS entidad_centralizada,
           count(*)                       AS n_contratos
    FROM stg_contratos WHERE codigo_entidad IS NOT NULL GROUP BY 1
),
desde_procesos AS (
    SELECT codigo_entidad,
           any_value(nombre_entidad)       AS nombre_entidad,
           any_value(nit_entidad)          AS nit_entidad,
           any_value(departamento_entidad) AS departamento,
           any_value(ciudad_entidad)       AS ciudad,
           any_value(orden_entidad)        AS orden,
           count(*)                        AS n_procesos
    FROM stg_procesos WHERE codigo_entidad IS NOT NULL GROUP BY 1
),
-- Ojo: datos_de_contacto contiene TODAS las cuentas de la plataforma
-- (entidades y proveedores). Solo se toman las marcadas como entidad.
claves AS (
    SELECT codigo_entidad FROM desde_contratos
    UNION SELECT codigo_entidad FROM desde_procesos
    UNION SELECT codigo_entidad FROM stg_contacto
      WHERE codigo_entidad IS NOT NULL AND upper(coalesce(es_entidad::VARCHAR,'')) = 'TRUE'
)
SELECT
    k.codigo_entidad,
    coalesce(ct.nombre_entidad, c.nombre_entidad, p.nombre_entidad)   AS nombre_entidad,
    norm(coalesce(ct.nombre_entidad, c.nombre_entidad, p.nombre_entidad)) AS nombre_entidad_norm,
    nitn(coalesce(c.nit_entidad, p.nit_entidad, ct.nit_entidad))      AS nit_entidad,
    norm(coalesce(c.departamento, p.departamento, ct.departamento))   AS departamento_norm,
    coalesce(c.ciudad, p.ciudad, ct.ciudad)                           AS ciudad,
    coalesce(c.orden, p.orden)                                        AS orden_entidad,
    c.sector,
    c.rama,
    c.entidad_centralizada,
    ct.tipo_entidad,
    ct.correo                                                         AS correo_entidad,
    ct.fax                                                            AS fax_entidad,
    ct.website,
    ct.codigo_ubicacion,
    ct.fecha_creacion                                                 AS fecha_registro,
    coalesce(ct.esta_activa, TRUE)                                    AS esta_activa,
    coalesce(c.n_contratos, 0)                                        AS n_contratos_hist,
    coalesce(p.n_procesos, 0)                                         AS n_procesos_hist
FROM claves k
LEFT JOIN desde_contratos c USING (codigo_entidad)
LEFT JOIN desde_procesos  p USING (codigo_entidad)
LEFT JOIN stg_contacto   ct USING (codigo_entidad);


-- --- dim_proveedor --------------------------------------------------------
-- Clave: codigo_proveedor. El TFM (cap. 5.1.1) justifica esta eleccion frente
-- al NIT, indefinido en ~51% de los registros de procesos.
CREATE OR REPLACE TABLE dim_proveedor AS
WITH desde_contratos AS (
    SELECT codigo_proveedor,
           any_value(nombre_proveedor)          AS nombre_proveedor,
           any_value(nombre_proveedor_canonico) AS nombre_canonico,
           any_value(nit_proveedor)             AS nit_proveedor,
           any_value(tipo_doc_proveedor)        AS tipo_doc,
           bool_or(es_grupo)                    AS es_grupo,
           bool_or(es_pyme)                     AS es_pyme,
           any_value(nombre_rep_legal)          AS nombre_rep_legal,
           any_value(doc_rep_legal)             AS doc_rep_legal,
           count(*)                             AS n_contratos
    FROM stg_contratos WHERE codigo_proveedor IS NOT NULL GROUP BY 1
),
claves AS (
    SELECT codigo_proveedor FROM desde_contratos
    UNION SELECT codigo_proveedor FROM stg_proveedores WHERE codigo_proveedor IS NOT NULL
    UNION SELECT codigo_proveedor FROM stg_procesos    WHERE codigo_proveedor IS NOT NULL
    UNION SELECT codigo_proveedor FROM stg_proponentes WHERE codigo_proveedor IS NOT NULL
)
SELECT
    k.codigo_proveedor,
    coalesce(r.nombre_proveedor, c.nombre_proveedor)                  AS nombre_proveedor,
    coalesce(r.nombre_proveedor_canonico, c.nombre_canonico)          AS nombre_canonico,
    nitn(coalesce(r.nit_proveedor, c.nit_proveedor))                  AS nit_proveedor,
    c.tipo_doc                                                        AS tipo_documento,
    coalesce(r.es_grupo, c.es_grupo, FALSE)                           AS es_grupo,
    coalesce(r.es_pyme, c.es_pyme)                                    AS es_pyme,
    coalesce(r.es_entidad, FALSE)                                     AS es_entidad_estatal,
    r.tipo_empresa,
    r.fecha_creacion                                                  AS fecha_registro,
    coalesce(r.esta_activa, TRUE)                                     AS esta_activa,
    norm(r.departamento)                                              AS departamento_norm,
    r.municipio,
    r.pais,
    r.codigo_categoria                                                AS codigo_categoria_principal,
    r.descripcion_categoria,
    -- Señales de contacto compartido: correo/telefono/fax repetidos entre
    -- proveedores distintos son indicador clasico de bid-rigging.
    r.correo,
    r.telefono,
    r.fax,
    r.sitio_web,
    coalesce(r.nombre_rep_legal, c.nombre_rep_legal)                  AS nombre_rep_legal,
    norm(coalesce(r.nombre_rep_legal, c.nombre_rep_legal))            AS nombre_rep_legal_norm,
    nitn(coalesce(r.doc_rep_legal, c.doc_rep_legal))                  AS doc_rep_legal,
    r.correo_rep_legal,
    coalesce(c.n_contratos, 0)                                        AS n_contratos_hist
FROM claves k
LEFT JOIN stg_proveedores  r USING (codigo_proveedor)
LEFT JOIN desde_contratos  c USING (codigo_proveedor);


-- --- dim_categoria (UNSPSC) ----------------------------------------------
CREATE OR REPLACE TABLE dim_categoria AS
WITH crudo AS (
    SELECT DISTINCT codigo_categoria FROM stg_contratos WHERE codigo_categoria IS NOT NULL
    UNION SELECT DISTINCT codigo_categoria FROM stg_procesos WHERE codigo_categoria IS NOT NULL
),
d AS (
    SELECT codigo_categoria,
           regexp_replace(codigo_categoria, '^V1\.', '') AS unspsc
    FROM crudo
)
SELECT
    codigo_categoria,
    unspsc                                  AS codigo_unspsc,
    substr(unspsc, 1, 2)                    AS segmento,
    substr(unspsc, 1, 4)                    AS familia,
    substr(unspsc, 1, 6)                    AS clase,
    CASE WHEN length(unspsc) >= 8 THEN unspsc END AS producto,
    length(unspsc)                          AS nivel_digitos
FROM d;


-- --- dim_modalidad / dim_tipo_contrato -----------------------------------
CREATE OR REPLACE TABLE dim_modalidad AS
WITH crudo AS (
    SELECT DISTINCT modalidad_de_contratacion AS m FROM stg_contratos WHERE modalidad_de_contratacion IS NOT NULL
    UNION SELECT DISTINCT modalidad_de_contratacion FROM stg_procesos WHERE modalidad_de_contratacion IS NOT NULL
)
SELECT
    m                                       AS modalidad_de_contratacion,
    norm(m)                                 AS modalidad_norm,
    -- agrupacion usada por el CRI de Fazekas (cap. 5.1.5 del TFM)
    CASE
      WHEN norm(m) LIKE '%DIRECTA%'            THEN 'Contratación directa'
      WHEN norm(m) LIKE '%MINIMA CUANT%'       THEN 'Mínima cuantía'
      WHEN norm(m) LIKE '%LICITACION%'         THEN 'Licitación pública'
      WHEN norm(m) LIKE '%ABREVIADA%'          THEN 'Selección abreviada'
      WHEN norm(m) LIKE '%CONCURSO DE MERITOS%' THEN 'Concurso de méritos'
      WHEN norm(m) LIKE '%REGIMEN ESPECIAL%'   THEN 'Régimen especial'
      ELSE 'Otra'
    END                                      AS grupo_modalidad,
    -- procedimientos sin concurrencia abierta
    norm(m) LIKE '%DIRECTA%'                 AS es_directa,
    norm(m) LIKE '%MINIMA CUANT%'            AS es_minima_cuantia,
    norm(m) LIKE '%LICITACION%' OR norm(m) LIKE '%ABREVIADA%'
                                             AS es_competitiva
FROM crudo;

CREATE OR REPLACE TABLE dim_tipo_contrato AS
WITH crudo AS (
    SELECT DISTINCT tipo_de_contrato AS t FROM stg_contratos WHERE tipo_de_contrato IS NOT NULL
    UNION SELECT DISTINCT tipo_de_contrato FROM stg_procesos WHERE tipo_de_contrato IS NOT NULL
)
SELECT t AS tipo_de_contrato,
       norm(t) AS tipo_norm,
       -- Obra e interventoria concentran el mayor riesgo colusorio (Wachs et al., 2021)
       norm(t) IN ('OBRA', 'INTERVENTORIA', 'CONSULTORIA') AS es_alto_riesgo
FROM crudo;


-- ==========================================================================
-- HECHOS
-- ==========================================================================

-- --- fact_contrato (GRANO PRINCIPAL) -------------------------------------
CREATE OR REPLACE TABLE fact_contrato AS
SELECT
    c.id_contrato,
    c.referencia_del_contrato,
    c.proceso_de_compra                                   AS id_portafolio,
    c.codigo_entidad,
    c.codigo_proveedor,
    c.codigo_categoria,
    c.modalidad_de_contratacion,
    c.tipo_de_contrato,
    c.fecha_firma,
    c.fecha_inicio,
    c.fecha_fin,
    CAST(strftime(c.fecha_firma, '%Y%m%d') AS INTEGER)    AS id_fecha_firma,
    year(c.fecha_firma)                                   AS anio_firma,
    date_diff('day', c.fecha_inicio, c.fecha_fin)         AS duracion_dias,
    c.estado_contrato,
    c.justificacion_modalidad,
    c.origen_de_los_recursos,
    c.destino_gasto,
    c.valor_del_contrato,
    c.valor_facturado,
    c.valor_pagado,
    c.valor_pendiente_de_pago,
    c.valor_pendiente_de_ejecucion,
    c.dias_adicionados,
    c.dias_adicionados > 0                                AS tiene_adiciones,
    c.liquidacion,
    c.es_postconflicto,
    c.es_grupo                                            AS proveedor_es_ut,
    c.es_pyme                                             AS proveedor_es_pyme,
    norm(c.departamento_entidad)                          AS departamento_norm,
    c.ciudad_entidad,
    c.sector,
    c.rama,
    c.orden_entidad,
    c.objeto_del_contrato,
    c.nombre_rep_legal,
    nitn(c.doc_rep_legal)                                 AS doc_rep_legal,
    c.nombre_supervisor,
    nitn(c.doc_supervisor)                                AS doc_supervisor,
    c.url_proceso,
    -- outlier: importes > 1 billon COP se descartan en el cap. 5.1.2 del TFM
    c.valor_del_contrato > 1e12                           AS es_outlier_valor
FROM stg_contratos c;


-- --- fact_proceso (base de todas las cifras del cap. 5 del TFM) ----------
CREATE OR REPLACE TABLE fact_proceso AS
SELECT
    p.id_del_proceso,
    p.referencia_del_proceso,
    p.id_portafolio,
    p.codigo_entidad,
    p.codigo_proveedor                                    AS codigo_proveedor_adjudicado,
    p.codigo_categoria,
    p.modalidad_de_contratacion,
    p.tipo_de_contrato,
    p.subtipo_de_contrato,
    p.fecha_publicacion,
    p.fecha_cierre_recepcion,
    p.fecha_adjudicacion,
    CAST(strftime(p.fecha_publicacion, '%Y%m%d') AS INTEGER) AS id_fecha_publicacion,
    year(p.fecha_publicacion)                             AS anio_publicacion,
    -- ventana de presentacion de ofertas: plazos muy cortos son bandera roja
    date_diff('day', p.fecha_publicacion, p.fecha_cierre_recepcion) AS dias_ventana_oferta,
    p.fase,
    p.estado_del_procedimiento,
    p.estado_resumen,
    p.estado_de_apertura_del_proceso,
    p.adjudicado,
    -- El campo 'adjudicado' de Socrata solo se marca 'Si' en una parte de los
    -- procesos: en 2025 hay 977.020 procesos en estado 'Seleccionado' pero
    -- solo 85.758 con adjudicado='Si'. Para analisis se usa este indicador,
    -- que reconoce tambien los procesos con adjudicatario o valor adjudicado.
    coalesce(p.adjudicado, FALSE)
      OR p.codigo_proveedor IS NOT NULL
      OR coalesce(p.valor_total_adjudicacion, 0) > 0               AS adjudicado_efectivo,
    p.precio_base,
    p.valor_total_adjudicacion,
    -- desviacion del valor adjudicado frente al presupuesto oficial
    CASE WHEN p.precio_base > 0 AND p.valor_total_adjudicacion > 0
         THEN (p.valor_total_adjudicacion - p.precio_base) / p.precio_base END AS ratio_desviacion_precio,
    p.duracion,
    p.unidad_de_duracion,
    p.numero_de_lotes,
    p.proveedores_invitados,
    p.proveedores_con_invitacion,
    p.proveedores_que_manifestaron,
    p.respuestas_al_procedimiento,
    p.conteo_de_respuestas_a_ofertas,
    p.proveedores_unicos_respuesta,
    p.visualizaciones,
    -- single bidder: señal primaria del CRI (27,2% de los procesos segun cap. 5.1.2)
    coalesce(p.proveedores_unicos_respuesta, 0) <= 1      AS es_single_bidder,
    norm(p.departamento_entidad)                          AS departamento_norm,
    p.ciudad_entidad,
    p.orden_entidad,
    norm(p.departamento_proveedor)                        AS departamento_proveedor_norm,
    p.nombre_del_procedimiento,
    p.descripcion_procedimiento,
    p.url_proceso,
    p.valor_total_adjudicacion > 1e12                     AS es_outlier_valor
FROM stg_procesos p;


-- --- fact_proponente (oferentes reales por proceso) ----------------------
CREATE OR REPLACE TABLE fact_proponente AS
SELECT
    pr.id_procedimiento                                   AS id_del_proceso,
    pr.codigo_proveedor,
    pr.codigo_entidad,
    pr.fecha_publicacion,
    year(pr.fecha_publicacion)                            AS anio,
    pr.nombre_procedimiento
FROM stg_proponentes pr
WHERE pr.id_procedimiento IS NOT NULL AND pr.codigo_proveedor IS NOT NULL;


-- --- fact_oferta (valor economico de cada oferta) ------------------------
CREATE OR REPLACE TABLE fact_oferta AS
SELECT
    o.id_del_proceso,
    o.id_oferta,
    o.codigo_proveedor,
    o.codigo_entidad,
    o.fecha_registro,
    year(o.fecha_registro)                                AS anio,
    o.valor_oferta,
    o.moneda,
    o.modalidad_de_contratacion,
    o.invitacion_directa,
    o.referencia_de_la_oferta,
    o.referencia_del_proceso
FROM stg_ofertas o;


-- --- fact_sancion --------------------------------------------------------
CREATE OR REPLACE TABLE fact_sancion AS
SELECT
    s.id_del_proceso,
    s.id_contrato,
    s.codigo_entidad,
    s.codigo_proveedor,
    s.nombre_proveedor,
    s.nombre_proveedor_canonico,
    s.fecha_evento,
    year(s.fecha_evento)                                  AS anio,
    s.valor_sancion,
    s.valor_pagado,
    s.tipo_de_sancion,
    s.tipo,
    s.estado,
    s.numero_de_acto,
    s.aplico_garantias
FROM stg_multas s;


-- --- bridge_ut_socio (red tripartita entidad-UT-socio) -------------------
CREATE OR REPLACE TABLE bridge_ut_socio AS
SELECT
    g.codigo_grupo                                        AS codigo_ut,
    g.nombre_grupo                                        AS nombre_ut,
    g.nombre_grupo_canonico,
    nitn(g.nit_grupo)                                     AS nit_ut,
    g.tipo_empresa_grupo                                  AS tipo_ut,
    g.fecha_creacion_grupo,
    g.esta_activo                                         AS ut_activa,
    norm(g.departamento_grupo)                            AS departamento_ut_norm,
    g.correo_grupo,
    g.telefono_grupo,
    nitn(g.doc_rep_legal_grupo)                           AS doc_rep_legal_ut,
    g.codigo_participante                                 AS codigo_socio,
    g.nombre_participante                                 AS nombre_socio,
    g.nombre_participante_canonico,
    nitn(g.nit_participante)                              AS nit_socio,
    g.tipo_empresa_participante                           AS tipo_socio,
    g.pct_participacion,
    g.es_lider,
    -- exclusion de banca/seguros: hubs artificiales, se descartan en cap. 5.1.4
    (norm(g.nombre_grupo)        LIKE '%SEGURO%' OR norm(g.nombre_participante) LIKE '%SEGURO%'
  OR norm(g.nombre_grupo)        LIKE '%ASEGURADORA%' OR norm(g.nombre_participante) LIKE '%ASEGURADORA%'
  OR norm(g.nombre_participante) LIKE '%FIDUCIARIA%' OR norm(g.nombre_grupo) LIKE '%FIDUCIARIA%'
  OR norm(g.nombre_participante) LIKE '%CORREDORES DE SEGUROS%'
  OR norm(g.nombre_participante) LIKE '%BANCO %')         AS es_banca_seguros
FROM stg_grupos g
WHERE g.codigo_grupo IS NOT NULL AND g.codigo_participante IS NOT NULL;


-- ==========================================================================
-- VISTAS ANALITICAS
-- ==========================================================================

-- --- Competencia real por proceso ----------------------------------------
CREATE OR REPLACE VIEW v_competencia_proceso AS
SELECT
    f.id_del_proceso,
    f.id_portafolio,
    f.codigo_entidad,
    f.anio_publicacion,
    f.modalidad_de_contratacion,
    f.tipo_de_contrato,
    f.precio_base,
    f.valor_total_adjudicacion,
    f.adjudicado,
    f.adjudicado_efectivo,
    f.dias_ventana_oferta,
    f.proveedores_unicos_respuesta            AS n_oferentes_declarado,
    coalesce(pp.n_proponentes, 0)             AS n_oferentes_real,
    coalesce(of.n_ofertas, 0)                 AS n_ofertas_con_valor,
    of.valor_min_oferta,
    of.valor_max_oferta,
    of.valor_med_oferta,
    -- dispersion de ofertas: valores muy proximos sugieren ofertas de cobertura
    CASE WHEN of.valor_med_oferta > 0
         THEN of.desv_oferta / of.valor_med_oferta END AS cv_ofertas,
    greatest(coalesce(pp.n_proponentes, 0),
             coalesce(f.proveedores_unicos_respuesta, 0)) <= 1 AS es_single_bidder
FROM fact_proceso f
LEFT JOIN (
    SELECT id_del_proceso, count(DISTINCT codigo_proveedor) AS n_proponentes
    FROM fact_proponente GROUP BY 1
) pp USING (id_del_proceso)
-- OJO con las claves: los proponentes cuelgan del procedimiento (CO1.REQ.*)
-- y las ofertas del proceso de compra o portafolio (CO1.BDOS.*). Son campos
-- distintos pese a llamarse igual en el origen, asi que las ofertas se unen
-- por id_portafolio, no por id_del_proceso.
LEFT JOIN (
    SELECT id_del_proceso AS id_portafolio,
           count(DISTINCT id_oferta) AS n_ofertas,
           min(valor_oferta)  AS valor_min_oferta,
           max(valor_oferta)  AS valor_max_oferta,
           avg(valor_oferta)  AS valor_med_oferta,
           stddev_samp(valor_oferta) AS desv_oferta
    FROM fact_oferta WHERE valor_oferta > 0 GROUP BY 1
) of USING (id_portafolio);


-- --- CRI de Fazekas por proveedor (Tabla 5.7 del TFM) --------------------
-- Se calcula sobre procesos efectivamente adjudicados, igual que en el TFM.
CREATE OR REPLACE VIEW v_cri_proveedor AS
WITH base AS (
    SELECT
        f.codigo_proveedor_adjudicado AS codigo_proveedor,
        f.codigo_entidad,
        f.es_single_bidder,
        coalesce(m.es_directa, FALSE)        AS es_directa,
        coalesce(m.es_minima_cuantia, FALSE) AS es_minima,
        f.valor_total_adjudicacion
    FROM fact_proceso f
    LEFT JOIN dim_modalidad m USING (modalidad_de_contratacion)
    WHERE f.adjudicado_efectivo AND f.codigo_proveedor_adjudicado IS NOT NULL
      AND NOT f.es_outlier_valor
),
agg AS (
    SELECT
        codigo_proveedor,
        count(*)                                        AS n_contratos,
        count(DISTINCT codigo_entidad)                  AS n_entidades,
        sum(valor_total_adjudicacion)                   AS valor_total,
        avg(valor_total_adjudicacion)                   AS valor_medio,
        avg(CASE WHEN es_single_bidder THEN 1.0 ELSE 0 END) AS pct_single,
        avg(CASE WHEN es_directa       THEN 1.0 ELSE 0 END) AS pct_directa,
        avg(CASE WHEN es_minima        THEN 1.0 ELSE 0 END) AS pct_minima
    FROM base GROUP BY 1
),
cri AS (
    SELECT *,
        1 - (n_entidades::DOUBLE / n_contratos)          AS concentracion_entidades,
        least(n_contratos / 50.0, 1.0)                   AS recurrencia
    FROM agg
)
SELECT *,
    0.30 * pct_single
  + 0.20 * pct_directa
  + 0.20 * concentracion_entidades
  + 0.15 * recurrencia
  + 0.15 * pct_minima                                    AS cri_score,
    0.30 * pct_single
  + 0.20 * pct_directa
  + 0.20 * concentracion_entidades
  + 0.15 * recurrencia
  + 0.15 * pct_minima
      >= quantile_cont(0.30 * pct_single + 0.20 * pct_directa
                     + 0.20 * concentracion_entidades + 0.15 * recurrencia
                     + 0.15 * pct_minima, 0.9) OVER ()   AS riesgo_alto
FROM cri;


-- --- Grafo bipartito entidad-proveedor (cap. 5.1.3) ----------------------
CREATE OR REPLACE VIEW v_arista_entidad_proveedor AS
SELECT
    codigo_entidad,
    codigo_proveedor_adjudicado AS codigo_proveedor,
    count(*)                    AS n_contratos,
    sum(valor_total_adjudicacion) AS valor_total,
    min(anio_publicacion)       AS primer_anio,
    max(anio_publicacion)       AS ultimo_anio
FROM fact_proceso
WHERE adjudicado_efectivo AND codigo_proveedor_adjudicado IS NOT NULL
  AND NOT es_outlier_valor
GROUP BY 1, 2;


-- --- Grafo de co-licitacion (proveedores que compiten en el mismo proceso)
-- Es la red estandar de bid-rigging; el TFM no pudo construirla por falta de
-- datos de proponentes y aqui queda disponible.
CREATE OR REPLACE VIEW v_arista_colicitacion AS
SELECT
    a.codigo_proveedor          AS proveedor_a,
    b.codigo_proveedor          AS proveedor_b,
    count(DISTINCT a.id_del_proceso) AS n_procesos_juntos,
    count(DISTINCT a.codigo_entidad) AS n_entidades_juntos
FROM fact_proponente a
JOIN fact_proponente b
  ON a.id_del_proceso = b.id_del_proceso
 AND a.codigo_proveedor < b.codigo_proveedor
GROUP BY 1, 2;


-- --- Triangulos de riesgo entidad-UT-socio (cap. 5.1.4) ------------------
CREATE OR REPLACE VIEW v_triangulo_riesgo AS
WITH adjudica AS (      -- entidad -> UT
    SELECT DISTINCT e.codigo_entidad, e.codigo_proveedor AS codigo_ut
    FROM v_arista_entidad_proveedor e
    JOIN (SELECT DISTINCT codigo_ut FROM bridge_ut_socio WHERE NOT es_banca_seguros) u
      ON u.codigo_ut = e.codigo_proveedor
),
integra AS (            -- UT -> socio
    SELECT DISTINCT codigo_ut, codigo_socio
    FROM bridge_ut_socio
    WHERE NOT es_banca_seguros
      -- La union temporal y el socio han de ser actores distintos: el puente
      -- contiene autolazos en que una UT figura como integrante de si misma, y
      -- sin esta condicion cada uno produce un «triangulo» de dos actores, que
      -- no es la estructura descrita en 5.1.4 sino una adjudicacion contada dos
      -- veces. La definicion canonica vive en tfm/grafo/triangulos.py.
      AND codigo_socio <> codigo_ut
),
directo AS (            -- socio -> entidad (contratacion por cuenta propia)
    SELECT DISTINCT codigo_entidad, codigo_proveedor AS codigo_socio
    FROM v_arista_entidad_proveedor
)
SELECT
    a.codigo_entidad,
    a.codigo_ut,
    i.codigo_socio
FROM adjudica a
JOIN integra i ON i.codigo_ut = a.codigo_ut
JOIN directo d ON d.codigo_entidad = a.codigo_entidad
              AND d.codigo_socio   = i.codigo_socio;


-- --- Señales de contacto compartido entre proveedores --------------------
-- Mismo correo / telefono / representante legal en proveedores distintos.
CREATE OR REPLACE VIEW v_senal_contacto_compartido AS
WITH campos AS (
    SELECT 'correo'        AS tipo_senal, correo               AS valor, codigo_proveedor FROM dim_proveedor WHERE correo IS NOT NULL
    UNION ALL SELECT 'telefono',          telefono,               codigo_proveedor FROM dim_proveedor WHERE telefono IS NOT NULL
    UNION ALL SELECT 'rep_legal_doc',     doc_rep_legal,          codigo_proveedor FROM dim_proveedor WHERE doc_rep_legal IS NOT NULL
    UNION ALL SELECT 'rep_legal_nombre',  nombre_rep_legal_norm,  codigo_proveedor FROM dim_proveedor WHERE nombre_rep_legal_norm IS NOT NULL
)
SELECT tipo_senal, valor,
       count(DISTINCT codigo_proveedor) AS n_proveedores,
       list(DISTINCT codigo_proveedor)  AS proveedores
FROM campos
GROUP BY 1, 2
HAVING count(DISTINCT codigo_proveedor) > 1;
