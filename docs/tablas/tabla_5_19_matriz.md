# Matriz de control: cuatro arquitecturas por dos conjuntos de entrada

Fuente: `OUTPUTS/tablas/tabla_5_19_matriz.csv`, generado por el análisis.

| Conjunto | Modelo | AUC-ROC | PR-AUC | F1 | Precision@200 | Recall@200 |
|:---|:---|---:|---:|---:|---:|---:|
| propias | SNA - Regresión Logística | 0,8076 | 0,2415 | 0,2971 | 0,455 | 0,0362 |
| propias | ML - Random Forest | 0,9051 | 0,4767 | 0,1835 | 0,745 | 0,0592 |
| propias | ML - Gradient Boosting | 0,9016 | 0,474 | 0,1947 | 0,78 | 0,062 |
| propias | SGC - propagación + MLP | 0,8586 | 0,3306 | 0,0654 | 0,585 | 0,0465 |
| propagadas | SNA - Regresión Logística | 0,8199 | 0,2597 | 0,0324 | 0,375 | 0,0298 |
| propagadas | ML - Random Forest | 0,942 | 0,662 | 0,3944 | 0,965 | 0,0767 |
| propagadas | ML - Gradient Boosting | 0,9329 | 0,6012 | 0,3597 | 0,885 | 0,0703 |
| propagadas | SGC - propagación + MLP | 0,9328 | 0,5862 | 0,4433 | 0,875 | 0,0696 |
