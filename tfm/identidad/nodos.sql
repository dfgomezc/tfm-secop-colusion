-- ==========================================================================
-- 05_nodos.sql — Universo de red para el TFM: contratos suscritos en 2025,
--                sin prestacion de servicios.
--
-- Lo ejecuta 05_nodos.py, que expone las vistas stg_* sobre bronze/ y las
-- tablas del modelo dimensional (gold/parquet).
--
-- Salida:
--   contrato_2025       un contrato firmado en 2025 (alcance del analisis)
--   proceso_2025        el procedimiento del que nace cada contrato
--   participacion_2025  nodo x proceso: inscrito / oferto / adjudicado
--   nodo_2025           tabla de atributos de nodo (SECOP + RUES)
--   vinculo_2025        aristas tipadas entre nodos
--   nodo_candidato_splink   atributos de bloqueo para la fase probabilistica
--
-- Taxonomia de identificadores del SECOP II (verificada sobre los datos):
--   CO1.REQ.*     procedimiento  -> proceso.id_del_proceso, proponentes
--   CO1.BDOS.*    proceso de compra / portafolio -> contrato, OFERTAS
--   CO1.PCCNTR.*  contrato
-- Las ofertas cuelgan del BDOS, los proponentes del REQ. No son la misma clave.
-- ==========================================================================

CREATE OR REPLACE MACRO norm(s) AS
    nullif(trim(regexp_replace(upper(strip_accents(CAST(s AS VARCHAR))),
                               '\s+', ' ', 'g')), '');

-- Documento a solo digitos.
CREATE OR REPLACE MACRO soldig(s) AS
    nullif(regexp_replace(CAST(s AS VARCHAR), '[^0-9]', '', 'g'), '');

-- Documento utilizable: descarta los centinelas basura detectados en la
-- exploracion (000000000 lo comparten 746 cuentas distintas, casi todas UT).
-- (RE2 no admite retrorreferencias: el "todo el mismo digito" se comprueba
--  comparando con la repeticion del primer caracter, mucho mas barato que
--  list_distinct(string_split(...)) sobre los 9,3 M de registros del RUES.)
CREATE OR REPLACE MACRO docok(s) AS
    CASE WHEN soldig(s) IS NULL THEN NULL
         WHEN length(soldig(s)) < 6 THEN NULL
         WHEN soldig(s) = repeat(substr(soldig(s), 1, 1), length(soldig(s))) THEN NULL
         WHEN soldig(s) IN ('123456789', '1234567890', '12345678',
                            '987654321', '999999999') THEN NULL
         ELSE soldig(s) END;


-- ==========================================================================
-- 0. Insumos del alcance
-- ==========================================================================

-- 0.1 Persona natural. Se combinan las dos señales disponibles en SECOP:
--     el tipo de documento con el que firma y la forma juridica registrada.
--     No coinciden del todo (846.507 contratos de 2025 con cedula frente a
--     723.462 con tipo_empresa 'PERSONA NATURAL COLOMBIANA'), asi que se toma
--     la union de ambas. El RUES arbitra despues en nodo_2025 via
--     organizacion_juridica.
CREATE OR REPLACE TABLE actor_persona_natural AS
SELECT
    p.codigo_proveedor,
    coalesce(
        p.tipo_documento IN ('Cédula de Ciudadanía', 'Cédula de Extranjería',
                             'Tarjeta de Identidad', 'Permiso por Protección Temporal',
                             'Permiso especial de permanencia')
     OR p.tipo_empresa = 'PERSONA NATURAL COLOMBIANA', FALSE)   AS persona_natural
FROM dim_proveedor p;

-- 0.2 Nucleo de red: actores que participan en la estructura relacional, por
--     integrar una union temporal o por haberse presentado a algun proceso.
--     Sirve para NO perder a las personas naturales que si forman parte de la
--     red (1.971 integran UT): de ellas interesa precisamente ver si ademas
--     reciben contratos directos.
CREATE OR REPLACE TABLE nucleo_red AS
SELECT codigo_socio      AS codigo_proveedor FROM bridge_ut_socio
UNION SELECT codigo_ut                        FROM bridge_ut_socio
UNION SELECT codigo_proveedor                 FROM fact_proponente
UNION SELECT codigo_proveedor                 FROM fact_oferta;


-- ==========================================================================
-- 1. ALCANCE: contratos suscritos entre 2025-01-01 y 2025-12-31.
--
--    Se excluye la prestacion de servicios de personas naturales, que en 2025
--    son 849.883 contratos de 542.668 personas: un volumen que domina el grafo
--    sin aportar estructura (contratacion directa, sin competencia y sin
--    uniones temporales).
--
--    EXCEPCION: si esa persona natural pertenece al nucleo de red, sus
--    contratos SI entran. Son 23.051 contratos de 11.980 personas, y son
--    justo el caso que interesa observar: el socio de una UT que ademas
--    contrata directamente por su cuenta.
--
--    Resultado: 224.025 contratos, 83.396 proveedores, 158,5 bill COP.
-- ==========================================================================
CREATE OR REPLACE TABLE contrato_2025 AS
SELECT
    c.id_contrato,
    c.referencia_del_contrato,
    c.id_portafolio,                              -- CO1.BDOS.* : une con proceso y ofertas
    c.codigo_entidad,
    c.codigo_proveedor,
    c.codigo_categoria,
    c.modalidad_de_contratacion,
    c.tipo_de_contrato,
    c.fecha_firma,
    c.fecha_inicio,
    c.fecha_fin,
    c.duracion_dias,
    c.estado_contrato,
    c.justificacion_modalidad,
    c.origen_de_los_recursos,
    c.destino_gasto,
    c.valor_del_contrato,
    c.valor_facturado,
    c.valor_pagado,
    c.valor_pendiente_de_ejecucion,
    c.dias_adicionados,
    c.tiene_adiciones,
    c.liquidacion,
    c.proveedor_es_ut,
    c.proveedor_es_pyme,
    c.departamento_norm,
    c.ciudad_entidad,
    c.sector,
    c.rama,
    c.orden_entidad,
    c.objeto_del_contrato,
    c.doc_rep_legal,
    c.nombre_rep_legal,
    c.doc_supervisor,
    c.nombre_supervisor,
    c.url_proceso,
    c.es_outlier_valor,
    coalesce(pn.persona_natural, FALSE)           AS proveedor_es_persona_natural,
    nr.codigo_proveedor IS NOT NULL               AS proveedor_en_nucleo_red
FROM fact_contrato c
LEFT JOIN actor_persona_natural pn USING (codigo_proveedor)
LEFT JOIN nucleo_red            nr USING (codigo_proveedor)
WHERE c.fecha_firma BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
  AND NOT (c.tipo_de_contrato = 'Prestación de servicios'
           AND coalesce(pn.persona_natural, FALSE)
           AND nr.codigo_proveedor IS NULL);


-- ==========================================================================
-- 2. Procesos de los que nacen esos contratos
-- ==========================================================================
CREATE OR REPLACE TABLE proceso_2025 AS
SELECT
    p.id_del_proceso,                             -- CO1.REQ.*
    p.id_portafolio,                              -- CO1.BDOS.*
    p.referencia_del_proceso,
    p.codigo_entidad,
    p.codigo_proveedor_adjudicado,
    p.codigo_categoria,
    p.modalidad_de_contratacion,
    p.tipo_de_contrato,
    p.fecha_publicacion,
    p.fecha_cierre_recepcion,
    p.fecha_adjudicacion,
    p.dias_ventana_oferta,
    p.fase,
    p.estado_del_procedimiento,
    p.adjudicado,
    p.adjudicado_efectivo,
    p.precio_base,
    p.valor_total_adjudicacion,
    p.ratio_desviacion_precio,
    p.numero_de_lotes,
    p.proveedores_invitados,
    p.proveedores_que_manifestaron,
    p.respuestas_al_procedimiento,
    p.proveedores_unicos_respuesta,
    p.visualizaciones,
    p.es_single_bidder                            AS single_bidder_declarado,
    p.departamento_norm,
    p.nombre_del_procedimiento,
    p.url_proceso
FROM fact_proceso p
WHERE EXISTS (SELECT 1 FROM contrato_2025 c WHERE c.id_portafolio = p.id_portafolio);


-- ==========================================================================
-- 3. PARTICIPACION: nodo x proceso, con la distincion inscrito / oferto
--    - proponentes  cuelgan del REQ  (se inscribieron)
--    - ofertas      cuelgan del BDOS (enviaron propuesta con valor)
-- ==========================================================================
CREATE OR REPLACE TABLE participacion_2025 AS
WITH prop AS (          -- inscritos en el procedimiento
    SELECT DISTINCT pr.codigo_proveedor, p.id_del_proceso, p.id_portafolio
    FROM proceso_2025 p
    JOIN fact_proponente pr ON pr.id_del_proceso = p.id_del_proceso
),
ofe AS (                -- enviaron oferta economica (join por BDOS)
    SELECT o.codigo_proveedor, o.id_del_proceso AS id_portafolio,
           count(DISTINCT o.id_oferta) AS n_ofertas,
           min(o.valor_oferta)  AS valor_oferta_min,
           max(o.valor_oferta)  AS valor_oferta_max,
           min(o.fecha_registro) AS fecha_primera_oferta
    FROM fact_oferta o
    WHERE EXISTS (SELECT 1 FROM proceso_2025 p WHERE p.id_portafolio = o.id_del_proceso)
    GROUP BY 1, 2
),
adj AS (                -- se le adjudico un contrato del alcance
    SELECT DISTINCT codigo_proveedor, id_portafolio, id_contrato, valor_del_contrato
    FROM contrato_2025
),
llaves AS (
    SELECT codigo_proveedor, id_portafolio FROM prop
    UNION SELECT codigo_proveedor, id_portafolio FROM ofe
    UNION SELECT codigo_proveedor, id_portafolio FROM adj
)
SELECT
    k.codigo_proveedor,
    k.id_portafolio,
    any_value(pr.id_del_proceso)                  AS id_del_proceso,
    pr.codigo_proveedor IS NOT NULL               AS es_proponente,
    o.codigo_proveedor  IS NOT NULL               AS presento_oferta,
    a.codigo_proveedor  IS NOT NULL               AS fue_adjudicado,
    coalesce(o.n_ofertas, 0)                      AS n_ofertas,
    o.valor_oferta_min,
    o.valor_oferta_max,
    o.fecha_primera_oferta,
    a.id_contrato,
    a.valor_del_contrato
FROM llaves k
LEFT JOIN prop pr USING (codigo_proveedor, id_portafolio)
LEFT JOIN ofe  o  USING (codigo_proveedor, id_portafolio)
LEFT JOIN adj  a  USING (codigo_proveedor, id_portafolio)
GROUP BY ALL;


-- ==========================================================================
-- 4. UNIVERSO DE NODOS
-- ==========================================================================
-- 4.1 Codigos de proveedor implicados: adjudicatarios, proponentes, oferentes
--     y ademas los socios de las UT adjudicatarias o participantes.
--     Ojo con el alcance: bridge_ut_socio cubre 2015-2026 y tiene 299.465 UT.
--     Arrastrar toda UT en la que alguien del alcance haya sido socio alguna
--     vez mete 239.600 nodos historicos sin actividad en 2025 y desvirtua la
--     red. Solo entran las UT que aparecen en el alcance de 2025; la
--     trayectoria historica se conserva como atributo (n_ut_que_integra).
CREATE OR REPLACE TABLE nodo_codigo AS
WITH directo AS (       -- adjudicatarios, proponentes y oferentes de 2025
    SELECT DISTINCT codigo_proveedor AS cod FROM participacion_2025
),
socios AS (             -- integrantes de las UT que aparecen en el alcance
    SELECT DISTINCT b.codigo_socio AS cod
    FROM bridge_ut_socio b
    WHERE b.codigo_ut IN (SELECT cod FROM directo)
),
uts AS (                -- UT del alcance en las que participa un socio del alcance
    SELECT DISTINCT b.codigo_ut AS cod
    FROM bridge_ut_socio b
    WHERE b.codigo_socio IN (SELECT cod FROM directo)
      AND b.codigo_ut    IN (SELECT cod FROM directo)
)
SELECT cod FROM directo WHERE cod IS NOT NULL
UNION SELECT cod FROM socios WHERE cod IS NOT NULL
UNION SELECT cod FROM uts   WHERE cod IS NOT NULL;


-- 4.2 Documento consolidado por codigo, y resolucion del digito de verificacion.
--     Regla conservadora: solo se quita el DV cuando el RUES lo confirma
--     (numero_identificacion + digito_verificacion = nit) o cuando el documento
--     sin DV existe en SECOP con el MISMO nombre canonico. La exploracion
--     mostro falsos positivos (900319113 LUCILA RICO vs 9003191131 BIOQUIMICA
--     META) si se recorta a ciegas.
CREATE OR REPLACE TABLE nodo_doc AS
-- Las uniones temporales no estan en proveedores_registrados, asi que
-- dim_proveedor no les da ni nombre ni NIT (42.973 UT quedaban sin nombre).
-- bridge_ut_socio si los trae: se usa como respaldo por ambos lados.
WITH ut_como_grupo AS (
    SELECT codigo_ut AS cod, max(nombre_ut) AS nombre,
           max(nombre_grupo_canonico) AS canonico, max(nit_ut) AS nit,
           max(tipo_ut) AS tipo_empresa, max(departamento_ut_norm) AS depto,
           max(correo_grupo) AS correo, max(telefono_grupo) AS telefono,
           max(doc_rep_legal_ut) AS doc_rl
    FROM bridge_ut_socio GROUP BY 1
),
ut_como_socio AS (
    SELECT codigo_socio AS cod, max(nombre_socio) AS nombre,
           max(nombre_participante_canonico) AS canonico, max(nit_socio) AS nit,
           max(tipo_socio) AS tipo_empresa
    FROM bridge_ut_socio GROUP BY 1
),
base AS (
    SELECT n.cod,
           docok(coalesce(p.nit_proveedor, gu.nit, gs.nit))  AS doc,
           coalesce(p.nombre_proveedor, gu.nombre, gs.nombre)     AS nombre_proveedor,
           coalesce(p.nombre_canonico, gu.canonico, gs.canonico)  AS nombre_canonico,
           p.tipo_documento,
           coalesce(p.tipo_empresa, gu.tipo_empresa, gs.tipo_empresa) AS tipo_empresa,
           coalesce(p.es_grupo, gu.cod IS NOT NULL, FALSE)        AS es_grupo,
           coalesce(p.departamento_norm, gu.depto)                AS departamento_norm,
           p.municipio,
           coalesce(p.correo, gu.correo)                          AS correo,
           coalesce(p.telefono, gu.telefono)                      AS telefono,
           p.nombre_rep_legal,
           docok(coalesce(p.doc_rep_legal, gu.doc_rl))            AS doc_rep_legal_secop,
           coalesce(pn.persona_natural, FALSE)                    AS persona_natural_secop
    FROM nodo_codigo n
    LEFT JOIN dim_proveedor p ON p.codigo_proveedor = n.cod
    LEFT JOIN actor_persona_natural pn ON pn.codigo_proveedor = n.cod
    LEFT JOIN ut_como_grupo gu ON gu.cod = n.cod
    LEFT JOIN ut_como_socio gs ON gs.cod = n.cod
),
-- El RUES arbitra: si el documento aparece como `nit` (con DV) y su
-- numero_identificacion es el mismo sin el ultimo digito, se usa ese.
arbitro_rues AS (
    SELECT DISTINCT docok(nit) AS doc_con_dv, docok(numero_identificacion) AS doc_base
    FROM stg_rues
    WHERE nit IS NOT NULL AND numero_identificacion IS NOT NULL
      AND docok(nit) = docok(numero_identificacion) || right(docok(nit), 1)
),
-- Alternativa: el documento recortado existe en SECOP con el mismo nombre.
arbitro_secop AS (
    SELECT DISTINCT a.doc AS doc_con_dv, b.doc AS doc_base
    FROM base a
    JOIN base b ON b.doc = substr(a.doc, 1, length(a.doc) - 1)
               AND b.nombre_canonico = a.nombre_canonico
    WHERE a.doc IS NOT NULL AND b.doc IS NOT NULL
)
SELECT
    b.cod,
    b.doc                                                   AS doc_declarado,
    coalesce(r.doc_base, s.doc_base, b.doc)                 AS doc_base,
    CASE WHEN r.doc_base IS NOT NULL THEN 'RUES'
         WHEN s.doc_base IS NOT NULL THEN 'nombre SECOP'
         WHEN b.doc IS NULL          THEN 'sin documento'
         ELSE 'sin cambio' END                              AS metodo_dv,
    b.nombre_proveedor,
    b.nombre_canonico,
    b.tipo_documento,
    b.tipo_empresa,
    b.es_grupo,
    b.departamento_norm,
    b.municipio,
    b.correo,
    b.telefono,
    b.nombre_rep_legal,
    b.doc_rep_legal_secop,
    b.persona_natural_secop
FROM base b
LEFT JOIN arbitro_rues  r ON r.doc_con_dv = b.doc
LEFT JOIN arbitro_secop s ON s.doc_con_dv = b.doc;


-- 4.3a RUES reducido: solo las matriculas cuyo documento aparece en el
--      universo de nodos, con doc_base precalculado. Baja de 9,3 M de filas a
--      las que de verdad hacen falta, y evita recalcular docok() mas de una vez.
CREATE OR REPLACE TABLE rues_util AS
SELECT r.*, d.doc_base
FROM stg_rues r
JOIN (SELECT DISTINCT doc_base FROM nodo_doc WHERE doc_base IS NOT NULL) d
  ON d.doc_base = docok(r.numero_identificacion);


-- 4.3b Mejor matricula del RUES por documento (la activa y mas reciente).
CREATE OR REPLACE TABLE rues_mejor AS
SELECT * EXCLUDE (_rn) FROM (
    SELECT
        doc_base,
        razon_social, razon_social_canonica, sigla,
        clase_identificacion, digito_verificacion,
        camara_comercio, codigo_camara,
        matricula, inscripcion_proponente,
        ciiu1, ciiu2, ciiu3, ciiu4,
        fecha_matricula, fecha_renovacion, ultimo_ano_renovado,
        fecha_cancelacion, estado_matricula,
        tipo_sociedad, organizacion_juridica, categoria_matricula,
        representante_legal, representante_legal_canonico,
        docok(doc_representante_legal)             AS doc_rep_legal_rues,
        clase_identificacion_rl,
        row_number() OVER (
            PARTITION BY doc_base
            ORDER BY CASE WHEN estado_matricula = 'ACTIVA' THEN 0 ELSE 1 END,
                     fecha_actualizacion DESC NULLS LAST,
                     fecha_matricula DESC NULLS LAST) AS _rn
    FROM rues_util
) WHERE _rn = 1;


-- 4.4 TABLA DE NODOS
--     id_nodo: clave analitica. Los actores con documento utilizable se unen
--     por documento base; los demas conservan su codigo de plataforma.
CREATE OR REPLACE TABLE nodo_2025 AS
WITH por_cod AS (          -- una fila por cuenta de plataforma, con su id_nodo
    SELECT d.*,
           coalesce('DOC:' || d.doc_base, 'COD:' || d.cod)   AS id_nodo
    FROM nodo_doc d
),
-- Rol en el alcance
rol AS (
    SELECT d.cod,
           bool_or(p.es_proponente)   AS fue_proponente,
           bool_or(p.presento_oferta) AS presento_oferta,
           bool_or(p.fue_adjudicado)  AS fue_adjudicado,
           count(DISTINCT p.id_portafolio) AS n_procesos
    FROM nodo_doc d LEFT JOIN participacion_2025 p ON p.codigo_proveedor = d.cod
    GROUP BY 1
),
-- Los dos conteos se calculan por separado: unirlos con dos LEFT JOIN sobre
-- bridge_ut_socio produce el producto cartesiano de ambos lados en los hubs.
como_socio AS (
    SELECT codigo_socio AS cod, count(DISTINCT codigo_ut) AS n_ut_integra
    FROM bridge_ut_socio GROUP BY 1
),
como_ut AS (
    SELECT codigo_ut AS cod, count(DISTINCT codigo_socio) AS n_socios_si_es_ut
    FROM bridge_ut_socio GROUP BY 1
),
ut AS (
    SELECT d.cod,
           coalesce(s.n_ut_integra, 0)       AS n_ut_integra,
           coalesce(u.n_socios_si_es_ut, 0)  AS n_socios_si_es_ut
    FROM nodo_doc d
    LEFT JOIN como_socio s ON s.cod = d.cod
    LEFT JOIN como_ut    u ON u.cod = d.cod
),
contratos AS (
    SELECT d.cod,
           count(DISTINCT c.id_contrato)              AS n_contratos_2025,
           sum(c.valor_del_contrato)                  AS valor_contratos_2025,
           count(DISTINCT c.codigo_entidad)           AS n_entidades_2025,
           count(DISTINCT c.id_contrato) FILTER (
               WHERE m.es_directa OR m.es_minima_cuantia)  AS n_contratos_directos_2025,
           sum(c.valor_del_contrato) FILTER (
               WHERE m.es_directa OR m.es_minima_cuantia)  AS valor_directo_2025
    FROM nodo_doc d
    LEFT JOIN contrato_2025 c ON c.codigo_proveedor = d.cod AND NOT c.es_outlier_valor
    LEFT JOIN dim_modalidad m USING (modalidad_de_contratacion)
    GROUP BY 1
),
-- Se agrega por id_nodo sobre TODAS las cuentas del actor, no solo la primera.
unido AS (
    SELECT p.*, o.* EXCLUDE (cod), u.* EXCLUDE (cod), c.* EXCLUDE (cod)
    FROM por_cod p
    LEFT JOIN rol       o USING (cod)
    LEFT JOIN ut        u USING (cod)
    LEFT JOIN contratos c USING (cod)
),
agrupado AS (
    SELECT
        id_nodo,
        doc_base                                             AS documento,
        list(DISTINCT cod)                                   AS codigos_proveedor,
        count(DISTINCT cod)                                  AS n_cuentas_plataforma,
        max(nombre_proveedor)                                AS nombre_secop,
        max(nombre_canonico)                                 AS nombre_canonico,
        max(tipo_documento)                                  AS tipo_documento,
        max(tipo_empresa)                                    AS tipo_empresa,
        bool_or(es_grupo)                                    AS es_grupo,
        bool_or(persona_natural_secop)                       AS persona_natural_secop,
        max(metodo_dv)                                       AS metodo_dv,
        max(departamento_norm)                               AS departamento_norm,
        max(municipio)                                       AS municipio,
        max(correo)                                          AS correo,
        max(telefono)                                        AS telefono,
        max(nombre_rep_legal)                                AS nombre_rep_legal_secop,
        max(doc_rep_legal_secop)                             AS doc_rep_legal_secop,
        coalesce(bool_or(fue_proponente), FALSE)             AS fue_proponente,
        coalesce(bool_or(presento_oferta), FALSE)            AS presento_oferta,
        coalesce(bool_or(fue_adjudicado), FALSE)             AS fue_adjudicado,
        coalesce(sum(n_procesos), 0)                         AS n_procesos_2025,
        coalesce(sum(n_contratos_2025), 0)                   AS n_contratos_2025,
        sum(valor_contratos_2025)                            AS valor_contratos_2025,
        coalesce(sum(n_entidades_2025), 0)                   AS n_entidades_2025,
        coalesce(sum(n_contratos_directos_2025), 0)          AS n_contratos_directos_2025,
        sum(valor_directo_2025)                              AS valor_directo_2025,
        coalesce(sum(n_ut_integra), 0)                       AS n_ut_que_integra,
        coalesce(sum(n_socios_si_es_ut), 0)                  AS n_socios
    FROM unido GROUP BY 1, 2
)
SELECT
    g.id_nodo,
    g.documento,
    g.codigos_proveedor,
    g.n_cuentas_plataforma,
    -- Tipo de nodo. El RUES arbitra cuando existe matricula; si no, se usan
    -- las señales de SECOP (documento de firma y forma juridica declarada).
    CASE WHEN g.es_grupo OR g.n_socios > 0                    THEN 'union_temporal'
         WHEN r.organizacion_juridica = 'PERSONA NATURAL'     THEN 'persona_natural'
         WHEN r.organizacion_juridica IS NOT NULL             THEN 'persona_juridica'
         WHEN g.persona_natural_secop                         THEN 'persona_natural'
         WHEN g.tipo_documento = 'NIT'                        THEN 'persona_juridica'
         WHEN g.tipo_documento IS NULL                        THEN 'sin_clasificar'
         ELSE 'persona_natural' END                           AS tipo_nodo,
    coalesce(r.razon_social, g.nombre_secop)                  AS nombre,
    g.nombre_canonico,
    g.tipo_documento,
    g.tipo_empresa,
    g.metodo_dv,
    -- Rol en el alcance 2025
    g.fue_proponente,
    g.presento_oferta,
    g.fue_adjudicado,
    g.n_procesos_2025,
    g.n_contratos_2025,
    g.valor_contratos_2025,
    g.n_entidades_2025,
    g.n_contratos_directos_2025,
    g.valor_directo_2025,
    g.n_ut_que_integra,
    g.n_socios,
    -- Contacto (señales de vinculo encubierto)
    g.departamento_norm,
    g.municipio,
    g.correo,
    g.telefono,
    -- Atributos RUES
    r.doc_base IS NOT NULL                                    AS cruza_rues,
    r.sigla,
    r.camara_comercio,
    r.ciiu1, r.ciiu2, r.ciiu3, r.ciiu4,
    r.fecha_matricula,
    date_diff('year', r.fecha_matricula, DATE '2025-01-01')    AS antiguedad_anios_2025,
    r.ultimo_ano_renovado,
    r.estado_matricula,
    r.fecha_cancelacion,
    r.tipo_sociedad,
    r.organizacion_juridica,
    r.categoria_matricula,
    r.inscripcion_proponente,
    coalesce(r.representante_legal, g.nombre_rep_legal_secop)  AS representante_legal,
    r.representante_legal_canonico,
    coalesce(r.doc_rep_legal_rues, g.doc_rep_legal_secop)      AS doc_representante_legal
FROM agrupado g
LEFT JOIN rues_mejor r ON r.doc_base = g.documento;


-- 4.4b Puente nodo -> cuenta de plataforma.
--      La columna codigos_proveedor de nodo_2025 es una lista, y al exportarse
--      a Parquet viaja como texto (Power BI no maneja listas). Esta tabla es la
--      forma relacional de recorrer lo mismo, y sirve para unir cualquier
--      tabla que este a nivel de codigo_proveedor.
CREATE OR REPLACE TABLE nodo_cuenta AS
SELECT coalesce('DOC:' || doc_base, 'COD:' || cod) AS id_nodo,
       cod                                          AS codigo_proveedor
FROM nodo_doc WHERE cod IS NOT NULL;


-- 4.5 Nodos de representante legal (entran al universo, como pediste)
CREATE OR REPLACE TABLE nodo_representante AS
WITH desde_rues AS (
    SELECT DISTINCT docok(n.doc_representante_legal) AS doc,
           n.representante_legal AS nombre
    FROM nodo_2025 n WHERE n.doc_representante_legal IS NOT NULL
),
desde_secop AS (
    SELECT DISTINCT docok(c.doc_rep_legal) AS doc,
           CAST(c.nombre_rep_legal AS VARCHAR) AS nombre
    FROM contrato_2025 c WHERE docok(c.doc_rep_legal) IS NOT NULL
),
todos AS (
    SELECT doc, nombre FROM desde_rues
    UNION ALL SELECT doc, nombre FROM desde_secop
)
SELECT 'RL:' || doc              AS id_nodo,
       doc                       AS documento,
       'representante_legal'     AS tipo_nodo,
       max(nombre)               AS nombre,
       norm(max(nombre))         AS nombre_canonico,
       -- cuantos actores distintos declara representar: la señal que interesa
       count(*)                  AS n_apariciones
FROM todos WHERE doc IS NOT NULL GROUP BY 1, 2, 3;


-- 4.6 Nodos entidad contratante
CREATE OR REPLACE TABLE nodo_entidad AS
SELECT 'ENT:' || e.codigo_entidad AS id_nodo,
       e.codigo_entidad,
       soldig(e.nit_entidad)      AS documento,
       'entidad_estatal'          AS tipo_nodo,
       e.nombre_entidad           AS nombre,
       e.nombre_entidad_norm      AS nombre_canonico,
       e.orden_entidad, e.sector, e.rama, e.entidad_centralizada,
       e.tipo_entidad, e.departamento_norm, e.ciudad,
       count(DISTINCT c.id_contrato)    AS n_contratos_2025,
       sum(c.valor_del_contrato)        AS valor_contratos_2025,
       count(DISTINCT c.codigo_proveedor) AS n_proveedores_2025
FROM dim_entidad e
JOIN contrato_2025 c ON c.codigo_entidad = e.codigo_entidad
GROUP BY ALL;


-- ==========================================================================
-- 4.7 Competencia por proceso, y tope para las aristas de co-participacion.
--
--     Las aristas de co-oferta son cuadraticas en el numero de oferentes. En
--     2025 hay 746 procesos con mas de 200 oferentes (el mayor, 2.903): son
--     acuerdos marco y catalogos, donde concurrir a la vez no significa
--     competir entre si. Sin tope generan 179 M de pares, el 99% del total,
--     y ahogan la señal de colusion.
--
--     Con tope de 100 oferentes quedan 1,75 M de pares sobre 48.233 procesos,
--     el 98,1% de los procesos con dos o mas oferentes.
-- ==========================================================================
CREATE OR REPLACE TABLE proceso_competencia AS
SELECT
    id_portafolio,
    count(*) FILTER (WHERE es_proponente)   AS n_proponentes,
    count(*) FILTER (WHERE presento_oferta) AS n_oferentes,
    count(*) FILTER (WHERE fue_adjudicado)  AS n_adjudicatarios,
    count(*) FILTER (WHERE presento_oferta) BETWEEN 2 AND 100 AS apto_co_oferta,
    count(*) FILTER (WHERE es_proponente)   BETWEEN 2 AND 100 AS apto_co_presentacion
FROM participacion_2025
GROUP BY 1;


-- ==========================================================================
-- 5. VINCULOS (aristas tipadas)
-- ==========================================================================
CREATE OR REPLACE TABLE vinculo_2025 AS
-- 5.1 entidad --adjudica--> nodo
--     Incluye tambien los contratos sin proceso competitivo; el campo detalle
--     distingue si la adjudicacion fue directa. Es lo que permite cerrar el
--     triangulo de riesgo: socio de una UT adjudicataria que ademas contrata
--     directamente con la misma entidad.
SELECT 'adjudica'                          AS tipo_vinculo,
       'ENT:' || c.codigo_entidad          AS origen,
       n.id_nodo                           AS destino,
       count(DISTINCT c.id_contrato)       AS peso,
       sum(c.valor_del_contrato)           AS valor,
       CASE WHEN bool_and(coalesce(m.es_directa, FALSE))  THEN 'solo directo'
            WHEN bool_or(coalesce(m.es_directa, FALSE))   THEN 'mixto'
            ELSE 'con competencia' END      AS detalle
FROM contrato_2025 c
JOIN nodo_doc d ON d.cod = c.codigo_proveedor
JOIN nodo_2025 n ON n.id_nodo = coalesce('DOC:' || d.doc_base, 'COD:' || d.cod)
LEFT JOIN dim_modalidad m USING (modalidad_de_contratacion)
WHERE NOT c.es_outlier_valor
GROUP BY 1, 2, 3

UNION ALL
-- 5.2 nodo --se_presenta_a--> entidad (inscrito, gane o no)
SELECT 'se_presenta_a', n.id_nodo, 'ENT:' || p.codigo_entidad,
       count(DISTINCT pa.id_portafolio), NULL,
       CASE WHEN bool_or(pa.presento_oferta) THEN 'con oferta' ELSE 'solo inscrito' END
FROM participacion_2025 pa
JOIN proceso_2025 p ON p.id_portafolio = pa.id_portafolio
JOIN nodo_doc d ON d.cod = pa.codigo_proveedor
JOIN nodo_2025 n ON n.id_nodo = coalesce('DOC:' || d.doc_base, 'COD:' || d.cod)
GROUP BY 1, 2, 3

UNION ALL
-- 5.3 union temporal --integra--> socio
SELECT 'integra', nu.id_nodo, ns.id_nodo,
       count(*), NULL,
       CASE WHEN bool_or(b.es_lider) THEN 'lider' END
FROM bridge_ut_socio b
JOIN nodo_doc du ON du.cod = b.codigo_ut
JOIN nodo_doc ds ON ds.cod = b.codigo_socio
JOIN nodo_2025 nu ON nu.id_nodo = coalesce('DOC:' || du.doc_base, 'COD:' || du.cod)
JOIN nodo_2025 ns ON ns.id_nodo = coalesce('DOC:' || ds.doc_base, 'COD:' || ds.cod)
GROUP BY 1, 2, 3

UNION ALL
-- 5.4 nodo --representado_por--> representante legal
SELECT 'representado_por', n.id_nodo, 'RL:' || n.doc_representante_legal,
       1, NULL, n.representante_legal
FROM nodo_2025 n WHERE n.doc_representante_legal IS NOT NULL

UNION ALL
-- 5.5 co-oferta: dos nodos que enviaron oferta al mismo proceso de compra
SELECT 'co_oferta', a.id_nodo, b.id_nodo,
       count(DISTINCT x.id_portafolio), NULL, NULL
FROM participacion_2025 x
JOIN proceso_competencia pc ON pc.id_portafolio = x.id_portafolio AND pc.apto_co_oferta
JOIN participacion_2025 y ON y.id_portafolio = x.id_portafolio
JOIN nodo_doc dx ON dx.cod = x.codigo_proveedor
JOIN nodo_doc dy ON dy.cod = y.codigo_proveedor
JOIN nodo_2025 a ON a.id_nodo = coalesce('DOC:' || dx.doc_base, 'COD:' || dx.cod)
JOIN nodo_2025 b ON b.id_nodo = coalesce('DOC:' || dy.doc_base, 'COD:' || dy.cod)
WHERE x.presento_oferta AND y.presento_oferta AND a.id_nodo < b.id_nodo
GROUP BY 1, 2, 3

UNION ALL
-- 5.6 co-presentacion: inscritos en el mismo proceso (aunque no oferten)
SELECT 'co_presentacion', a.id_nodo, b.id_nodo,
       count(DISTINCT x.id_portafolio), NULL, NULL
FROM participacion_2025 x
JOIN proceso_competencia pc ON pc.id_portafolio = x.id_portafolio AND pc.apto_co_presentacion
JOIN participacion_2025 y ON y.id_portafolio = x.id_portafolio
JOIN nodo_doc dx ON dx.cod = x.codigo_proveedor
JOIN nodo_doc dy ON dy.cod = y.codigo_proveedor
JOIN nodo_2025 a ON a.id_nodo = coalesce('DOC:' || dx.doc_base, 'COD:' || dx.cod)
JOIN nodo_2025 b ON b.id_nodo = coalesce('DOC:' || dy.doc_base, 'COD:' || dy.cod)
WHERE x.es_proponente AND y.es_proponente AND a.id_nodo < b.id_nodo
GROUP BY 1, 2, 3

UNION ALL
-- 5.7 comparte_contacto: mismo correo o telefono entre nodos distintos.
--     Vinculo encubierto clasico entre supuestos competidores. Se descartan
--     los valores compartidos por mas de 20 nodos (centralitas, gestorias).
SELECT 'comparte_contacto', a.id_nodo, b.id_nodo,
       count(*), NULL, any_value(a.canal)
FROM (
    SELECT id_nodo, 'correo'   AS canal, correo   AS valor FROM nodo_2025 WHERE correo   IS NOT NULL
    UNION ALL
    SELECT id_nodo, 'telefono' AS canal, telefono AS valor FROM nodo_2025 WHERE telefono IS NOT NULL
) a
JOIN (
    SELECT id_nodo, 'correo'   AS canal, correo   AS valor FROM nodo_2025 WHERE correo   IS NOT NULL
    UNION ALL
    SELECT id_nodo, 'telefono' AS canal, telefono AS valor FROM nodo_2025 WHERE telefono IS NOT NULL
) b ON b.valor = a.valor AND b.canal = a.canal AND a.id_nodo < b.id_nodo
WHERE a.valor IN (
    SELECT valor FROM (
        SELECT correo   AS valor FROM nodo_2025 WHERE correo   IS NOT NULL
        UNION ALL
        SELECT telefono        FROM nodo_2025 WHERE telefono IS NOT NULL
    ) GROUP BY 1 HAVING count(DISTINCT valor) >= 1 AND count(*) BETWEEN 2 AND 20
)
GROUP BY 1, 2, 3;


-- ==========================================================================
-- 6. Candidatos para la fase probabilistica (Splink)
--    Nodos sin documento utilizable o sin cruce con RUES: son los que hay que
--    intentar unir por razon social, representante legal y ubicacion.
-- ==========================================================================
CREATE OR REPLACE TABLE nodo_candidato_splink AS
SELECT
    n.id_nodo,
    n.documento,
    n.nombre,
    n.nombre_canonico,
    -- claves de bloqueo habituales
    substr(n.nombre_canonico, 1, 4)                          AS bloque_prefijo,
    regexp_replace(n.nombre_canonico, '[^A-Z]', '', 'g')     AS solo_letras,
    length(n.nombre_canonico)                                AS long_nombre,
    n.tipo_nodo,
    n.camara_comercio,
    n.ciiu1,
    n.representante_legal_canonico,
    n.doc_representante_legal,
    n.fecha_matricula,
    n.documento IS NULL                                      AS sin_documento,
    NOT n.cruza_rues                                         AS sin_rues
FROM nodo_2025 n
WHERE n.documento IS NULL OR NOT n.cruza_rues;


-- ==========================================================================
-- 7. Triangulo de riesgo, ya sobre el universo acotado:
--    un socio que integra una UT adjudicataria de una entidad y que ADEMAS
--    contrata directamente con esa misma entidad. Es la pregunta del TFM.
-- ==========================================================================
CREATE OR REPLACE VIEW v_triangulo_2025 AS
WITH adjudica AS (
    SELECT origen AS entidad, destino AS nodo, peso AS contratos, valor, detalle
    FROM vinculo_2025 WHERE tipo_vinculo = 'adjudica'
),
integra AS (
    SELECT origen AS nodo_ut, destino AS nodo_socio
    FROM vinculo_2025 WHERE tipo_vinculo = 'integra'
)
SELECT
    a.entidad,
    a.nodo                     AS nodo_ut,
    i.nodo_socio,
    a.contratos                AS contratos_ut,
    a.valor                    AS valor_ut,
    d.contratos                AS contratos_directos_socio,
    d.valor                    AS valor_directo_socio,
    d.detalle                  AS modalidad_socio
FROM adjudica a
JOIN integra  i ON i.nodo_ut = a.nodo
JOIN adjudica d ON d.entidad = a.entidad AND d.nodo = i.nodo_socio;
