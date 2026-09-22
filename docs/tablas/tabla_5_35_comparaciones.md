# Diferencia emparejada de AUC-ROC entre pares de modelos

Fuente: `OUTPUTS/tablas/tabla_5_35_comparaciones.csv`, generado por el análisis.

| Comparación | Diferencia observada | IC 95% inferior | IC 95% superior | Remuestreos a favor | Afirmable |
|:---|---:|---:|---:|---:|:---|
| ML — Random Forest sobre variables propagadas − GNN — GraphSAGE con muestreo | 0,0064 | 0,0031 | 0,0098 | 1,0 | sí |
| GNN — GraphSAGE con muestreo − SGC — propagación + MLP | 0,0027 | -0,0005 | 0,0059 | 0,9445 | no |
| SGC — propagación + MLP − GNN — GCN de dos capas | 0,0098 | 0,006 | 0,0135 | 1,0 | sí |
| SGC — propagación + MLP − ML — Random Forest | 0,0277 | 0,024 | 0,0315 | 1,0 | sí |
