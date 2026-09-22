# Detección de riesgo de colusión en la contratación pública colombiana

Comparación entre técnicas clásicas de análisis de redes y redes neuronales de
grafos para priorizar riesgo de colusión sobre el Sistema Electrónico para la
Contratación Pública (SECOP II) de Colombia.

El repositorio contiene el código que produce cada cifra, tabla y figura del
estudio, el conjunto de datos seudonimizado que sirve de insumo a los modelos y
las tablas de resultados exportadas a markdown.

## Qué hace

A partir de los datos abiertos de contratación de 2025 se construye un grafo
tipado de actores económicos y entidades contratantes, se resuelve la identidad
de los actores, se calculan variables de red y se comparan nueve ordenamientos
de riesgo sobre la misma partición y la misma etiqueta:

| Familia | Ordenamientos |
|:---|:---|
| Sin entrenamiento | PageRank, recuento de banderas rojas estructurales, número de contratos |
| Tabular | Regresión logística, bosque aleatorio, gradient boosting |
| Relacional | SGC con propagación de características y perceptrón multicapa |
| Paso de mensajes | Red convolucional de grafos de dos capas, GraphSAGE con muestreo |

La etiqueta es una pseudo-etiqueta derivada del Índice de Riesgo de Corrupción
de Fazekas, tomando el percentil 90 como umbral. El estudio incluye el control
de fuga entre etiqueta y variables, la separación entre el efecto de la
arquitectura y el de la información de entrada, la cuantificación de la
incertidumbre por remuestreo y la explicación de las alertas por valores de
Shapley.

## Resultados

Las tablas del estudio están en [`docs/tablas/`](docs/tablas/README.md), en
markdown, y en `OUTPUTS/tablas/` en CSV. Las cifras principales:

| Ordenamiento | AUC-ROC | PR-AUC | Precision@200 |
|:---|---:|---:|---:|
| Bosque aleatorio sobre variables propagadas | 0,9420 | 0,6620 | 0,965 |
| GraphSAGE con muestreo | 0,9355 | 0,5924 | 0,870 |
| SGC con propagación y perceptrón | 0,9328 | 0,5862 | 0,875 |
| Red convolucional de grafos | 0,9230 | 0,5417 | 0,760 |
| Bosque aleatorio | 0,9051 | 0,4767 | 0,745 |
| Gradient boosting | 0,9016 | 0,4740 | 0,780 |
| Número de contratos | 0,8015 | 0,2409 | 0,450 |
| Regresión logística | 0,8076 | 0,2415 | 0,455 |
| PageRank | 0,7785 | 0,2190 | 0,385 |
| Banderas rojas estructurales | 0,4507 | 0,0970 | 0,025 |

Sobre el conjunto de prueba sin fuga, 24.896 actores con 2.516 positivos. La
amplitud media del intervalo de confianza al 95 % del AUC-ROC es de ocho
milésimas, de modo que las diferencias menores que esa cifra no se afirman: el
remuestreo de 2.000 réplicas establece que GraphSAGE y el SGC empatan sobre
este corpus.

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/dfgomezc/tfm-secop-colusion.git
cd tfm-secop-colusion
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python comprobar_entorno.py
```

`comprobar_entorno.py` verifica el intérprete, las dependencias, el árbol de
directorios y qué etapas pueden ejecutarse con los datos disponibles.

## Los datos

### `datos/`, versionado, 21 MB

Es el insumo del análisis y viaja en el repositorio. Los actores están
identificados por seudónimo, nunca por documento de identidad ni razón social.

| Carpeta | Contenido |
|:---|:---|
| `datos/red/features_nodo/` | Las 22 variables de red por actor y la etiqueta, 82.984 filas. Es la entrada directa de los modelos |
| `datos/red/banderas_rojas.csv` | Las cuatro banderas estructurales por actor |
| `datos/red/topologia_*.csv` | Descriptores del grafo bipartito y de la red tripartita |
| `datos/red/ego_redes_triangulos.csv` | Estadísticas estructurales de las dos ego-redes de mayor concentración |
| `datos/grafo/nodo_2025/` | Los 139.548 actores del universo con sus atributos no identificativos |
| `datos/grafo/vinculo_2025/` | Las 2.914.940 aristas del grafo, con su tipo y peso |
| `datos/curvas/` | Series de las curvas ROC, precisión-exhaustividad y lift |
| `datos/tablas/` | Las 28 tablas publicadas, en CSV |

Con este conjunto se reentrenan los nueve modelos, se recalculan todas las
métricas, se recuentan los triángulos de riesgo y se redibujan las figuras de
red y de resultados.

### `INPUT/`, no versionado

Los JSONL de origen: unos 58 GB repartidos en novecientos ficheros, descargados
del portal de datos abiertos del SECOP II y del Registro Único Empresarial y
Social. No hacen falta para reproducir los resultados, porque el análisis parte
de los parquet, y por su tamaño quedan fuera del control de versiones. Solo se
necesitan para rehacer la ingesta desde cero. `INPUT/README.md` documenta los
nueve conjuntos y su procedencia.

### `OUTPUTS/`, no versionado

El modelo dimensional completo y los artefactos intermedios de las ejecuciones:
1,9 GB de parquet más las rejillas, predicciones y puntos de control de los
modelos. Se reconstruye con el código. Los artefactos publicados que sí caben
en el repositorio están duplicados en `datos/`.

### Protección de datos

El universo incluye personas naturales contratistas y representantes legales.
Los datos de contratación son públicos por la Ley 1712 de 2014, pero el
tratamiento de datos personales se rige por la Ley 1581 de 2012, y publicar un
documento de identidad asociado a una puntuación de riesgo no es una
publicación de datos abiertos.

Por eso el conjunto distribuido está seudonimizado. `tfm/datos_publicables.py`
es el módulo que lo deriva y documenta cada decisión: sustituye los
identificadores por seudónimos estables, asigna seudónimo propio a los
representantes legales de modo que la estructura de representación compartida se
conserve, y elimina nombre, documento, correo, teléfono, objeto del contrato e
identificadores de proceso, que permitirían recuperar al actor desde el portal.

La sal criptográfica de la seudonimización y la clave de correspondencia entre
seudónimo y actor real no forman parte del repositorio.

## Uso

### Desde la línea de órdenes

```bash
python orquestador.py --listar               # etapas y datos disponibles
python orquestador.py --desde comparativa    # desde una etapa concreta
python orquestador.py --solo figuras
```

Las etapas se declaran en `orquestador.py` con los datos que cada una necesita,
de modo que una ejecución que no pueda completarse lo advierte antes de
empezar.

### Desde los cuadernos

```bash
jupyter lab notebooks/
```

| # | Cuaderno | Produce |
|--:|:---|:---|
| 00 | `inventario_y_balance` | Recuento de los datos disponibles y su cobertura temporal |
| 01 | `bronze_y_gold` | Modelo dimensional desde los JSONL |
| 02 | `resolucion_de_identidad` | Actores resueltos y pesos de evidencia |
| 03 | `grafo_y_features` | Variables de red e índice de riesgo |
| 04 | `comparativa_de_modelos` | Los cuatro modelos sobre la misma partición |
| 05 | `control_de_fuga` | Variantes V1 y V2 y matriz de arquitecturas |
| 06 | `brazo_clasico` | Ordenamientos clásicos y banderas rojas |
| 07 | `gnn_paso_mensajes` | Red convolucional de grafos y GraphSAGE |
| 08 | `solapamiento_y_metricas` | Solapamiento, lift y calibración |
| 09 | `figuras_y_tablas` | Figuras y tablas del estudio |

Los cuadernos 01 y 02 solo hacen falta para reconstruir el modelo dimensional
desde los JSONL. Partiendo de `datos/` puede empezarse por el 03.

### Regenerar las figuras y las tablas

```bash
python -m tfm.figuras.generales         # distribuciones y métricas
python -m tfm.figuras.red               # estructura de red
python -m tfm.figuras.convergencia      # convergencia de las dos GNN
python -m tfm.figuras.explicabilidad    # atribución de las alertas
python -m tfm.tablas_markdown           # docs/tablas/*.md desde los CSV
```

Las diecisiete figuras del estudio y las 28 tablas se producen con estas
órdenes. Ninguna está dibujada ni transcrita a mano.

## Estructura

```
tfm/
  rutas.py              resolución de rutas del proyecto
  datos.py              acceso al modelo dimensional sobre parquet
  datos_publicables.py  derivación del conjunto seudonimizado
  tablas_markdown.py    exportación de las tablas a markdown
  comparar.py           contraste entre una ejecución y los resultados publicados
  ingesta/              JSONL -> bronze -> modelo dimensional
  identidad/            resolución determinista y probabilística de actores
  grafo/                variables de red, índice de riesgo y triángulos
  modelos/              comparativa, control de fuga, brazo clásico, GNN,
                        explicabilidad e incertidumbre
  figuras/              generación de figuras desde las tablas de resultados
  verificacion/         invariantes de integridad y análisis de sensibilidad
notebooks/              los cuadernos, en orden de ejecución
datos/                  el conjunto seudonimizado que sirve de insumo
docs/tablas/            las tablas del estudio en markdown
```

## Reproducibilidad

Semilla 42 en todo el proceso. La partición es estratificada 70/30 y es la misma
para todos los modelos: 58.088 actores en entrenamiento y 24.896 en prueba, con
2.516 positivos.

Las dos redes con paso de mensajes están implementadas sobre NumPy, con la
retropropagación escrita a mano y contrastada contra diferencias finitas
centradas. No hay dependencia de una biblioteca de aprendizaje profundo ni de
una GPU.

### Versiones de las librerías

`requirements.txt` acota scikit-learn a la rama 1.4 a 1.6, y la razón es
concreta. El control de fuga ajusta una regresión lineal sobre las once
variables de la variante V0, y esa matriz es numéricamente singular: `strength`
coincide con `n_contratos`, `grado` con `n_entidades`, y además entran los
logaritmos de esas mismas magnitudes. Su número de condición es del orden de
10¹⁶, con lo que la solución de mínimos cuadrados no es única y cada versión de
la librería puede elegir una distinta según el controlador que use por debajo.

Está comprobado: con scikit-learn 1.9 los cinco R² lineales de la tabla de
reconstruibilidad bajan entre dos y ocho milésimas, y la fracción reconstruible
del índice pasa de 0,5194 a 0,4286. Resolviendo el mismo sistema con
`numpy.linalg.lstsq` se recuperan exactamente los valores publicados.

### Sobre el orden de las filas

Un motor de base de datos no se compromete a devolver las filas de una unión en
ningún orden concreto, y `tfm.grafo.features_sna` además desactiva
`preserve_insertion_order` para acotar el consumo de memoria. Ese orden acababa
escrito en el parquet, y de él dependían dos cosas: la partición, porque
`train_test_split` permuta índices posicionales, y el resultado de Louvain, que
aunque va sembrado depende del orden de inserción de los nodos.

El `COPY` que escribe el fichero lleva `ORDER BY id_nodo`, de modo que dos
regeneraciones dan el mismo parquet. Ese orden canónico se fijó después de
obtener los resultados publicados: volver a ejecutar la etapa de red produce un
`features_nodo.parquet` válido pero distinto del distribuido, y con él cifras
distintas. Quien quiera reproducir el estudio parte del parquet de `datos/`;
quien rehaga el análisis desde los JSONL obtendrá su propio universo y debe
contrastarlo como tal.

Leer un parquet ya escrito sí es determinista, comprobado con DuckDB de uno a
dieciséis hilos.

## Alcance de los resultados

Los datos proceden del Sistema Electrónico para la Contratación Pública,
operado por Colombia Compra Eficiente y publicados como datos abiertos al
amparo de la Ley 1712 de 2014.

Lo que el sistema identifica es **riesgo estructural**: patrones relacionales
que la literatura asocia a acuerdos entre oferentes. No identifica colusión
probada, que exige una decisión de la autoridad de competencia. La salida es
una priorización de auditorías y no tiene valor probatorio.

La etiqueta es un índice de riesgo construido sobre volumen y modalidad de
contratación, no un registro de casos sancionados. La comparación entre modelos
mide, por tanto, qué familia de técnicas captura mejor la estructura que ese
índice resume. Contrastar las alertas contra decisiones firmes de la autoridad
de competencia queda como trabajo pendiente.

## Licencia

MIT. Véase [LICENSE](LICENSE).
