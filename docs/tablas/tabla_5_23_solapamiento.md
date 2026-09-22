# Solapamiento entre la lista corta del modelo y la señal estructural

Fuente: `OUTPUTS/tablas/tabla_5_23_solapamiento.csv`, generado por el análisis.

| Configuración | Top-K del modelo | Actores en triángulo (prueba) | Intersección | Jaccard | % del top-K que está en triángulo | % de los de triángulo que están en el top-K | ambos | solo modelo | solo triángulo | ninguno | esperados si independientes | chi2 | p (chi2) | odds ratio | p (Fisher) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|
| V0 | 200 | 872 | 1 | 0,0009 | 0,5 | 0,11 | 1 | 199 | 871 | 23825 | 7,01 | 4,52 | 0,0335 | 0,137 | 0,0112 |
| V1 | 200 | 872 | 1 | 0,0009 | 0,5 | 0,11 | 1 | 199 | 871 | 23825 | 7,01 | 4,52 | 0,0335 | 0,137 | 0,0112 |
| V1-RF | 200 | 872 | 0 | 0,0 | 0,0 | 0,0 | 0 | 200 | 872 | 23824 | 7,01 | 6,31 | 0,012 | 0,0 | 0,00149 |
