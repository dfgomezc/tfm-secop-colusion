# Configuración local

El contenido de este directorio está excluido del control de versiones, salvo
este fichero. Aquí vive el material que no puede publicarse.

## `sal_seudonimo.txt`

Cadena aleatoria de 32 bytes que se concatena al identificador de cada actor
antes de aplicar HMAC-SHA256 para obtener su seudónimo. La genera
`tfm.identidad.seudonimizar` en la primera ejecución, con `secrets` y no con
`random`, de modo que no pueda reproducirse desde una semilla conocida.

Es lo que hace efectiva la seudonimización. El algoritmo es público y está en
el propio código: sin la sal, conocerlo no permite recomputar la
correspondencia entre seudónimo y actor real.

Una consecuencia práctica: si la sal se pierde, los seudónimos de una
ejecución posterior no coincidirán con los del conjunto distribuido. Conviene
conservarla junto al resto del material reservado del estudio, no dentro del
repositorio.

## Qué pasa si no está

`tfm.identidad.seudonimizar` la crea. `tfm.datos_publicables` no: se detiene y
lo indica, porque generar una sal nueva produciría un conjunto de datos cuyos
seudónimos no casan con los ya publicados.
