# Datos de origen

Esta carpeta recibe los JSONL descargados del portal de datos abiertos. No se
versiona: son unos 58 GB en novecientos y pico ficheros, son datos abiertos
redescargables y **no hacen falta para reproducir los resultados**, que parten
de los parquet seudonimizados de `datos/`.

Solo se necesita poblarla para rehacer la ingesta desde cero, es decir, para
reconstruir el modelo dimensional a partir del dato bruto.

## Los nueve conjuntos

| Carpeta esperada | Fuente | Contenido |
|:---|:---|:---|
| `procesos_contratacion_s2/` | SECOP II | Procesos de contratación, con su modalidad y estado |
| `contratos_electronicos_s2/` | SECOP II | Contratos electrónicos suscritos |
| `proponentes_por_proceso_s2/` | SECOP II | Quién se presentó a cada procedimiento |
| `ofertas_por_proceso_s2/` | SECOP II | Ofertas registradas por proceso de compra |
| `proveedores_registrados_s2/` | SECOP II | Registro de proveedores de la plataforma |
| `grupos_proveedores_s2/` | SECOP II | Composición de uniones temporales y consorcios |
| `datos_de_contacto_s2/` | SECOP II | Correo y teléfono declarados por los actores |
| `multas_sanciones_s2/` | SECOP II | Multas y sanciones registradas |
| `matriculas_rues/` | RUES | Matrículas del Registro Único Empresarial y Social |

Los ocho primeros se publican en el portal de datos abiertos del Estado
colombiano y se descargan por la interfaz de Socrata. El noveno procede del
Registro Único Empresarial y Social y actúa de árbitro en la resolución de
identidad de los actores.

## Cómo se disponen

Una carpeta por conjunto, con los `.jsonl` dentro. La ingesta lee cada carpeta
por lotes de ficheros para acotar el consumo de memoria, de modo que el reparto
en ficheros es indiferente.

```
INPUT/
  procesos_contratacion_s2/
    parte_0001.jsonl
    parte_0002.jsonl
    ...
  contratos_electronicos_s2/
    ...
```

Si los datos viven en otro disco, no hace falta copiarlos: basta con apuntar
ahí la variable de entorno `TFM_INPUT`.

```bash
export TFM_INPUT=/datos/secop          # Linux o macOS
$env:TFM_INPUT = "D:\datos\secop"      # PowerShell
```

`python -m tfm.rutas` imprime desde dónde va a leer una ejecución.

## Ingesta

```bash
python -m tfm.ingesta.bronze     # JSONL -> parquet tipado
python -m tfm.ingesta.gold       # parquet -> modelo dimensional
```

La primera etapa normaliza los valores centinela que el portal emplea en lugar
de nulos, tipa fechas, importes e indicadores, y calcula el nombre canónico que
después usa la resolución de identidad. La segunda construye el modelo
dimensional sobre el que trabaja todo el análisis.
