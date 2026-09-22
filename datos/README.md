# Conjunto de datos seudonimizado

El insumo del análisis. Con lo que hay en esta carpeta se reentrenan los nueve
ordenamientos, se recalculan todas las métricas, se recuentan los triángulos de
riesgo y se redibujan las figuras del estudio, sin descargar nada.

Lo genera `tfm/datos_publicables.py` a partir del modelo dimensional completo.

## Contenido

| Ruta | Filas | Contenido |
|:---|---:|:---|
| `red/features_nodo/` | 82.984 | Las 22 variables de red por actor más la etiqueta. Es la entrada directa de los modelos |
| `red/banderas_rojas.csv` | 82.984 | Las cuatro banderas estructurales por actor y su recuento normalizado |
| `red/solapamiento_actores.csv` | 24.896 | Zona de solapamiento de cada actor del conjunto de prueba |
| `red/topologia_grafo.csv` | — | Descriptores del grafo bipartito completo |
| `red/topologia_tripartita.csv` | — | Descriptores de la red tripartita entidad-UT-socio |
| `red/ego_redes_triangulos.csv` | — | Estadísticas estructurales de las dos ego-redes de mayor concentración |
| `grafo/nodo_2025/` | 139.548 | Los actores del universo con sus atributos no identificativos |
| `grafo/vinculo_2025/` | 2.914.940 | Las aristas del grafo, con tipo, peso y valor |
| `curvas/` | — | Series de las curvas ROC, precisión-exhaustividad y lift |
| `tablas/` | — | Las 28 tablas publicadas, en CSV |

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
python -m tfm.verificacion.invariantes
```

Recorre el conjunto y valida los recuentos, las claves y la coherencia entre
tablas.
