# Conjunto de datos seudonimizado

El insumo del análisis. Con lo que hay en esta carpeta se reentrenan los nueve
ordenamientos, se recalculan todas las métricas, se recuentan los triángulos de
riesgo y se redibujan las figuras del estudio, sin descargar nada.

Lo genera `tfm/datos_publicables.py` a partir del modelo dimensional completo.

## Cómo se usa

La estructura reproduce la de `OUTPUTS/`, de modo que el código trabaja con
este conjunto sin adaptar nada:

```bash
export TFM_OUTPUTS=datos             # PowerShell: $env:TFM_OUTPUTS = "datos"
python -m tfm.modelos.brazo_clasico
python -m tfm.figuras.red
```

## Contenido

| Ruta | Filas | Contenido |
|:---|---:|:---|
| `graph_sna/features_nodo.parquet` | 82.984 | Las 22 variables de red por actor más la etiqueta. Es la entrada directa de los modelos |
| `graph_sna/banderas_rojas.csv` | 82.984 | Las cuatro banderas estructurales por actor y su recuento normalizado |
| `graph_sna/solapamiento_actores.csv` | 24.896 | Zona de solapamiento de cada actor del conjunto de prueba |
| `graph_sna/topologia_*.csv` | — | Descriptores del grafo bipartito y de la red tripartita |
| `graph_sna/ego_redes_triangulos.csv` | — | Estadísticas de las dos ego-redes de mayor concentración |
| `gold/nodo_2025/` | 139.548 | Los actores del universo con sus atributos no identificativos |
| `gold/nodo_entidad/` | 3.872 | Las entidades contratantes |
| `gold/nodo_cuenta/` | 146.565 | Puente entre la cuenta de plataforma y el actor resuelto |
| `gold/contrato_2025/` | 223.987 | Los contratos del alcance |
| `gold/proceso_competencia/` | 217.363 | Oferentes y proponentes por proceso |
| `gold/vinculo_2025/` | 2.914.940 | Las aristas del grafo, con tipo, peso y valor |
| `gold/fact_contrato/` | 1.050.857 | Fechas de firma de 2025, para el recuento del universo de partida |
| `gold/splink_parametros/` | 29 | Pesos en bits de cada señal del modelo de identidad |
| `modelos/` | — | Predicciones, particiones, rejillas, atribuciones y puntos de control |
| `curvas/` | — | Series de las curvas ROC, precisión-exhaustividad y lift |
| `tablas/` | — | Las 28 tablas publicadas, en CSV |
| `figuras/` | — | Las 17 figuras del estudio, en PDF |

## Los identificadores

Cada actor aparece bajo un seudónimo con prefijo por tipo:

| Prefijo | Tipo de actor |
|:---|:---|
| `E-` | Entidad contratante |
| `P-` | Persona jurídica |
| `UT-` | Unión temporal o consorcio |
| `N-` | Persona natural |
| `R-` | Representante legal |
| `X-` | Actor sin clasificar |

El seudónimo es estable: el mismo actor lleva el mismo en todos los ficheros, de
modo que las tablas se pueden unir por él. Y es consistente: dos empresas que
comparten representante legal comparten también el seudónimo `R-`, con lo que la
estructura de representación compartida que el análisis explota se conserva
intacta.

El correlativo se asigna ordenando por el HMAC del identificador real, no
alfabéticamente ni por puntuación de riesgo, para que el propio seudónimo no
filtre información sobre el actor al que designa. La sal criptográfica no forma
parte del repositorio, así que la correspondencia no puede recomputarse aunque
se conozca el algoritmo.

## Lo que no está

Se han eliminado el documento de identidad, el nombre y la razón social, el
nombre canónico, el correo, el teléfono, el municipio, los códigos de la
plataforma, el número del Registro Único de Proponentes, el nombre y documento
de representantes legales y supervisores, el objeto del contrato y las URL de
los procesos.

Las tres últimas exclusiones merecen explicación: el objeto del contrato es
texto libre que con frecuencia nombra a la persona contratada, y un identificador
de proceso o una URL permiten recuperar el nombre del actor desde el portal, lo
que anularía la seudonimización por completo.

Todo lo demás se conserva sin modificar. Las magnitudes, las fechas, las
variables de red, las puntuaciones y las clasificaciones son las mismas cifras
con las que se obtuvieron los resultados publicados.

## Una advertencia sobre el orden de las filas

`graph_sna/features_nodo.parquet` conserva el orden físico de filas del fichero
original, y eso no es cosmético: `train_test_split` permuta índices
posicionales, de modo que reordenar el fichero produce otro conjunto de prueba
y con él otras métricas para todo lo que se evalúe sobre él. Si se regenera
este conjunto, hay que copiarlo fila a fila y no reconstruirlo con una consulta
ordenada.

## Formato

Parquet con compresión zstd, repartido en ficheros `part_NNNN.parquet` de menos
de 20 MB por carpeta. El reparto es por tamaño y el contenido es idéntico al de
un fichero único: todo el código lee estas carpetas con un comodín, de modo que
DuckDB junta las partes al vuelo.

```sql
SELECT count(*) FROM 'datos/grafo/vinculo_2025/*.parquet';
```

## Comprobación

```bash
export TFM_OUTPUTS=datos
python -m tfm.verificacion.invariantes
```

Reconoce que está comprobando el conjunto distribuido y valida los recuentos y
las claves de las tablas que lo forman, omitiendo las comprobaciones que
necesitan el modelo dimensional entero.

Y la comprobación que de verdad importa: recalcular una tabla y contrastarla
con la publicada.

```bash
python -m tfm.modelos.brazo_clasico
diff salidas/modelos/tabla_5_21_siete_ordenamientos.csv \
     datos/tablas/tabla_5_21_siete_ordenamientos.csv
```

Trece de las tablas publicadas se recalculan desde este conjunto y salen
idénticas byte a byte.
