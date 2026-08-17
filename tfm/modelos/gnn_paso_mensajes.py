#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GCN y GraphSAGE con paso de mensajes sobre el grafo bipartito.

El modelo relacional de la comparativa es una Simplified Graph Convolution: la
propagación se precomputa con productos matriciales y el resultado alimenta a un
clasificador. Aquí se implementan las dos arquitecturas que la SGC simplifica,
para medir qué aporta cada pieza:

  - la GCN de Kipf y Welling (2017), que es la SGC con las no linealidades
    intermedias;
  - el GraphSAGE de Hamilton et al. (2017), con agregador aprendido, muestreo
    del vecindario y concatenación con la representación propia.

Se propaga sobre el grafo bipartito sin proyectar —proveedor, entidad,
proveedor—, 143.360 aristas, y no sobre la proyección `A = B·Bᵀ`, que sobre los
mismos nodos tiene 23,9 millones: la proyección fabrica un vínculo por cada
comprador compartido y sobrepondera a las entidades concentradoras.

La retropropagación está escrita sobre NumPy y SciPy, sin biblioteca de
aprendizaje profundo, y se contrasta contra diferencias finitas centradas
(`--fase gradiente`). A esta escala el coste es de segundos y el resultado no
depende del orden de las sumas atómicas de una GPU.

Uso:
    python -m tfm.modelos.gnn_paso_mensajes --fase gradiente
    python -m tfm.modelos.gnn_paso_mensajes --fase entrenar --modelo gcn
    python -m tfm.modelos.gnn_paso_mensajes --fase entrenar --modelo sage
    python -m tfm.modelos.gnn_paso_mensajes --fase tabla

El entrenamiento guarda punto de control cada pocas épocas y se reanuda solo:
ejecutarlo varias veces seguidas continúa desde donde lo dejó.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from tfm import rutas

SEMILLA = 42
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = rutas.raiz()
MODELOS = rutas.resultados()
SALIDA = os.path.join(MODELOS, "gnn")

#: Las siete variables que sobreviven al control de fuga del apartado 5.2.2.
#: Se entrena sobre V1 y no sobre V0 para que la comparación con el resto de
#: la Tabla 5.21 sea homogénea: medir una arquitectura nueva sobre la
#: configuración con fuga mediría sobre todo la fuga.
FEATURES_V0 = [
    "grado", "strength", "pagerank", "en_componente_gigante",
    "n_contratos", "n_entidades", "log_n_contratos", "log_valor_total",
    "log_valor_medio", "n_procesos", "n_cuentas_plataforma",
]
RECONSTRUIBLES = ["n_contratos", "n_entidades", "log_n_contratos", "grado"]
FEATURES_V1 = [f for f in FEATURES_V0 if f not in RECONSTRUIBLES]

ETIQUETAS = {"gcn": "GNN — GCN de 2 capas",
             "sage": "GNN — GraphSAGE con muestreo"}


# ---------------------------------------------------------------------------
# Datos y grafo
# ---------------------------------------------------------------------------
def cargar():
    """Réplica exacta de `10_fuga_variantes.cargar()`.

    Se reproduce el mismo orden de filas y el mismo `dropna` porque la
    partición se obtiene después con la misma semilla, y basta que el orden
    difiera en una fila para que los conjuntos de entrenamiento y prueba dejen
    de coincidir con los del resto de la comparativa.
    """
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    ruta = rutas.artefacto("features_nodo.parquet").replace("\\", "/")
    df = con.execute(f"SELECT * FROM read_parquet('{ruta}')").df()
    pq = rutas.parquet_de("vinculo_2025", "*.parquet").replace("\\", "/")
    aristas = con.execute(f"SELECT origen, destino FROM read_parquet('{pq}') "
                          "WHERE tipo_vinculo = 'adjudica'").fetchall()
    con.close()
    return df.dropna(subset=FEATURES_V0).reset_index(drop=True), aristas


def incidencia(ids, aristas):
    """Matriz de incidencia proveedor × entidad, sin proyectar.

    Devuelve dos operadores normalizados por grado: `Pe` lleva mensajes de
    proveedor a entidad y `Pp` los devuelve de entidad a proveedor. Componerlos
    equivale a un salto en la proyección, pero sin materializarla.
    """
    idx = {n: i for i, n in enumerate(ids)}
    ent, filas, cols = {}, [], []
    for e, n in aristas:
        if n not in idx:
            continue
        filas.append(idx[n])
        cols.append(ent.setdefault(e, len(ent)))
    B = sparse.csr_matrix((np.ones(len(filas), dtype=np.float64),
                           (filas, cols)), shape=(len(ids), len(ent)))
    gp = np.asarray(B.sum(1)).ravel()
    ge = np.asarray(B.sum(0)).ravel()
    gp[gp == 0] = 1.0
    ge[ge == 0] = 1.0
    Pp = sparse.diags(1.0 / gp) @ B          # entidad -> proveedor
    Pe = sparse.diags(1.0 / ge) @ B.T        # proveedor -> entidad
    return B.tocsr(), Pp.tocsr(), Pe.tocsr()


# ---------------------------------------------------------------------------
# Piezas diferenciables
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(z, 0.0)


def d_relu(z):
    return (z > 0).astype(z.dtype)


def estable_log_sigmoide(z):
    """log σ(z) sin desbordar para |z| grande.

    La forma directa `np.where(z >= 0, -log1p(exp(-z)), z - log1p(exp(z)))`
    parece estable y no lo es: `np.where` evalúa **las dos ramas** antes de
    elegir, de modo que `exp(z)` desborda para z grande aunque ese valor se
    descarte. El resultado es correcto pero llena la salida de avisos. Aquí se
    calcula sobre el argumento ya acotado al semieje que corresponde.
    """
    neg = -np.abs(z)
    return np.minimum(z, 0.0) - np.log1p(np.exp(neg))


def perdida(z, y, peso_pos):
    """Entropía cruzada binaria con reponderación de la clase positiva.

    El desbalance es de uno a diez, y el apartado 5.2.4 exige tratarlo igual
    en todos los modelos de la comparativa. `peso_pos` es el análogo del
    `class_weight='balanced'` que exploran los modelos de scikit-learn.
    """
    w = np.where(y == 1, peso_pos, 1.0)
    lp = estable_log_sigmoide(z)
    ln = estable_log_sigmoide(-z)
    return -np.sum(w * (y * lp + (1 - y) * ln)) / np.sum(w)


def d_perdida(z, y, peso_pos):
    w = np.where(y == 1, peso_pos, 1.0)
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return w * (s - y) / np.sum(w)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class RedDeGrafos:
    """GCN o GraphSAGE de dos saltos sobre el grafo bipartito.

    Ambas comparten esqueleto y se diferencian en la regla de agregación, que
    es exactamente donde los dos artículos difieren:

    * `gcn` (Kipf y Welling, 2017): la representación del nodo se **suma** a
      la media del vecindario antes de la no linealidad, que es el efecto del
      lazo propio en `Â = D^{-1/2}(A+I)D^{-1/2}`.
    * `sage` (Hamilton et al., 2017): la representación propia y la media del
      vecindario se **concatenan** y se proyectan con matrices distintas, el
      vecindario se **muestrea** con abanico fijo y la salida de cada capa se
      normaliza en norma L2.

    La diferencia con la SGC del apartado 5.2 es que aquí hay una no linealidad
    entre los dos saltos y las matrices de peso se aprenden por descenso de
    gradiente, en lugar de promediar y entregar el resultado a un clasificador.
    """

    def __init__(self, d_ent, oculta, modo, rng, dtype=np.float32):
        self.modo = modo
        self.oculta = oculta
        self.dtype = dtype
        #: Inicialización de Glorot: mantiene la varianza de la señal entre
        #: capas y es la que usan los dos artículos de referencia.
        def glorot(m, n):
            lim = np.sqrt(6.0 / (m + n))
            return rng.uniform(-lim, lim, size=(m, n)).astype(dtype)

        if modo == "gcn":
            self.W1 = glorot(d_ent, oculta)
            self.W2 = glorot(oculta, oculta)
            self.w3 = glorot(oculta, 1)
        else:
            #: GraphSAGE proyecta por separado lo propio y lo agregado.
            self.W1 = glorot(d_ent, oculta)       # vecindario, capa 1
            self.W1p = glorot(d_ent, oculta)      # propio, capa 1
            self.W2 = glorot(oculta, oculta)      # vecindario, capa 2
            self.W2p = glorot(oculta, oculta)     # propio, capa 2
            self.w3 = glorot(oculta, 1)
        self.b1 = np.zeros(oculta, dtype=dtype)
        self.b2 = np.zeros(oculta, dtype=dtype)
        self.b3 = np.zeros(1, dtype=dtype)

    # -- utilidades de estado -------------------------------------------
    def nombres(self):
        base = ["W1", "W2", "w3", "b1", "b2", "b3"]
        return base + (["W1p", "W2p"] if self.modo == "sage" else [])

    def parametros(self):
        return {k: getattr(self, k) for k in self.nombres()}

    def cargar_parametros(self, d):
        for k in self.nombres():
            setattr(self, k, d[k])

    # -- paso hacia delante ---------------------------------------------
    def adelante(self, X, Pp, Pe, guardar=True):
        """Dos saltos: proveedor → entidad → proveedor, dos veces.

        Un salto en el grafo bipartito no conecta a dos proveedores: hace
        falta ir a la entidad y volver. Por eso cada «capa» son dos
        multiplicaciones dispersas, y dos capas dan el mismo alcance de dos
        saltos que la SGC precomputa con `Â` y `Â²`.
        """
        c = {}
        #: Capa 1
        He1 = Pe @ X                       # mensaje agregado en la entidad
        Ag1 = Pp @ He1                     # devuelto al proveedor
        if self.modo == "gcn":
            Z1 = (X + Ag1) @ self.W1 + self.b1
        else:
            Z1 = Ag1 @ self.W1 + X @ self.W1p + self.b1
        H1 = relu(Z1)
        if self.modo == "sage":
            n1 = np.linalg.norm(H1, axis=1, keepdims=True)
            n1[n1 == 0] = 1.0
            H1n = H1 / n1
        else:
            n1, H1n = None, H1

        #: Capa 2
        He2 = Pe @ H1n
        Ag2 = Pp @ He2
        if self.modo == "gcn":
            Z2 = (H1n + Ag2) @ self.W2 + self.b2
        else:
            Z2 = Ag2 @ self.W2 + H1n @ self.W2p + self.b2
        H2 = relu(Z2)
        if self.modo == "sage":
            n2 = np.linalg.norm(H2, axis=1, keepdims=True)
            n2[n2 == 0] = 1.0
            H2n = H2 / n2
        else:
            n2, H2n = None, H2

        z = (H2n @ self.w3 + self.b3).ravel()
        if guardar:
            c.update(X=X, Ag1=Ag1, Z1=Z1, H1=H1, n1=n1, H1n=H1n,
                     Ag2=Ag2, Z2=Z2, H2=H2, n2=n2, H2n=H2n)
        return z, c

    # -- paso hacia atrás ------------------------------------------------
    def atras(self, c, dz, Pp, Pe):
        """Gradientes analíticos. `dz` es ∂L/∂z sobre los nodos evaluados."""
        g = {}
        dz = dz.reshape(-1, 1)
        g["w3"] = c["H2n"].T @ dz
        g["b3"] = dz.sum(axis=0)
        dH2n = dz @ self.w3.T

        def d_normaliza(dHn, H, n):
            """Retropropaga a través de h / ‖h‖.

            La derivada no es `dHn / n`: normalizar acopla las componentes,
            y omitir el término de proyección es el error clásico en una
            implementación a mano de GraphSAGE.
            """
            if n is None:
                return dHn
            prod = np.sum(dHn * H, axis=1, keepdims=True)
            return dHn / n - H * prod / (n ** 3)

        dH2 = d_normaliza(dH2n, c["H2"], c["n2"])
        dZ2 = dH2 * d_relu(c["Z2"])
        if self.modo == "gcn":
            g["W2"] = (c["H1n"] + c["Ag2"]).T @ dZ2
            g["b2"] = dZ2.sum(axis=0)
            dEntrada2 = dZ2 @ self.W2.T
            dAg2 = dEntrada2
            dH1n = dEntrada2.copy()
        else:
            g["W2"] = c["Ag2"].T @ dZ2
            g["W2p"] = c["H1n"].T @ dZ2
            g["b2"] = dZ2.sum(axis=0)
            dAg2 = dZ2 @ self.W2.T
            dH1n = dZ2 @ self.W2p.T
        #: Ag = Pp·(Pe·H) es lineal, así que su transpuesta es Peᵀ·(Ppᵀ·dAg).
        dH1n += Pe.T @ (Pp.T @ dAg2)

        dH1 = d_normaliza(dH1n, c["H1"], c["n1"])
        dZ1 = dH1 * d_relu(c["Z1"])
        if self.modo == "gcn":
            g["W1"] = (c["X"] + c["Ag1"]).T @ dZ1
        else:
            g["W1"] = c["Ag1"].T @ dZ1
            g["W1p"] = c["X"].T @ dZ1
        g["b1"] = dZ1.sum(axis=0)
        return g


def juego_de_muestras(B, abanico, cuantas=8):
    """Precalcula varios grafos muestreados y los rota durante el entrenamiento.

    Muestrear en cada época costaba más que la época entera: son 82.984 filas
    recorridas en Python. Rotar entre ocho muestreos distintos conserva el
    efecto regularizador —el modelo no ve siempre el mismo vecindario— a coste
    despreciable, y el resultado sigue siendo determinista porque cada muestreo
    se genera con su propia semilla derivada de la global.
    """
    return [muestrear(B, np.random.default_rng(SEMILLA + 1000 + k), abanico)
            for k in range(cuantas)]


def muestrear(B, rng, abanico):
    """Submuestreo del vecindario al estilo de GraphSAGE.

    Hamilton et al. (2017) muestrean un número fijo de vecinos por nodo en vez
    de agregar todo el vecindario, y lo justifican por coste. Aquí el coste no
    aprieta, pero el muestreo tiene un segundo efecto que sí importa en este
    corpus: **acota la influencia de las entidades concentradoras**, que
    adjudican a miles de proveedores y dominarían cualquier media.
    """
    B = B.tocsr()
    filas, cols = [], []
    for i in range(B.shape[0]):
        ini, fin = B.indptr[i], B.indptr[i + 1]
        vecinos = B.indices[ini:fin]
        if len(vecinos) > abanico:
            vecinos = rng.choice(vecinos, size=abanico, replace=False)
        filas.extend([i] * len(vecinos))
        cols.extend(vecinos)
    M = sparse.csr_matrix((np.ones(len(filas), dtype=np.float32),
                           (filas, cols)), shape=B.shape)
    gp = np.asarray(M.sum(1)).ravel()
    ge = np.asarray(M.sum(0)).ravel()
    gp[gp == 0] = 1.0
    ge[ge == 0] = 1.0
    return ((sparse.diags(1.0 / gp) @ M).tocsr().astype(np.float32),
            (sparse.diags(1.0 / ge) @ M.T).tocsr().astype(np.float32))


# ---------------------------------------------------------------------------
# Métricas, idénticas a las del resto de la comparativa
# ---------------------------------------------------------------------------
def precision_en_k(y, p, k=200):
    return float(y[np.argsort(-p)[:k]].mean())


def recall_en_k(y, p, k=200):
    return float(y[np.argsort(-p)[:k]].sum() / max(y.sum(), 1))


def metricas(y, p, umbral=0.5):
    return {
        "AUC-ROC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "F1": f1_score(y, (p >= umbral).astype(int), zero_division=0),
        "Precision@200": precision_en_k(y, p),
        "Recall@200": recall_en_k(y, p),
    }


# ---------------------------------------------------------------------------
def preparar():
    """Variables normalizadas, etiqueta, partición y operadores del grafo."""
    df, aristas = cargar()
    X = df[FEATURES_V1].astype(float).values
    y = df["riesgo_alto"].astype(int).values
    #: Estandarización con los estadísticos del conjunto de entrenamiento, no
    #: de todo el corpus: usar la media global filtraría información de prueba.
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA, stratify=y)
    #: De ese 70 % se aparta un 20 % de validación para decidir cuándo parar y
    #: qué configuración gana. El conjunto de prueba no interviene en ninguna
    #: decisión: se mira una sola vez, al final. Los modelos de scikit-learn de
    #: la comparativa hacen lo equivalente con validación cruzada de tres
    #: pliegues sobre el mismo 70 %.
    i_aj, i_val = train_test_split(i_tr, test_size=0.20,
                                   random_state=SEMILLA, stratify=y[i_tr])
    mu, sd = X[i_aj].mean(0), X[i_aj].std(0)
    sd[sd == 0] = 1.0
    #: Precisión simple: duplica la velocidad de los productos densos, que son
    #: el 90 % del coste, y no altera ninguna métrica a cuatro decimales. La
    #: comprobación de gradiente sí trabaja en doble precisión, porque allí el
    #: redondeo de la diferencia finita sería mayor que el error que se busca.
    Xn = ((X - mu) / sd).astype(np.float32)
    B, Pp, Pe = incidencia(df["id_nodo"].tolist(), aristas)
    return (df, Xn, y, i_aj, i_val, i_te, B,
            Pp.astype(np.float32), Pe.astype(np.float32))


def ruta_estado(modo):
    return os.path.join(SALIDA, f"_estado_{modo}.npz")


#: Rejilla de hiperparámetros. El apartado 5.2.4 reprocha a la versión anterior
#: haber explorado ocho combinaciones en unos modelos y cuatro en otros; medir
#: una arquitectura nueva con una sola configuración repetiría ese defecto en
#: su forma más grave. Son cuatro combinaciones, las mismas para las dos
#: arquitecturas, y la elección se hace sobre validación.
#:
#: Las anchuras exploradas son 16 y 32 y no 32 y 64 por dos razones. La
#: sustantiva es que la entrada tiene siete variables: una capa de 64 unidades
#: multiplica por nueve la dimensión de la señal, y el sondeo previo no mostró
#: ninguna ganancia por ese lado. La práctica es que el coste crece con el
#: cuadrado de la anchura y la máquina disponible tiene dos núcleos, de modo
#: que 64 unidades habrían consumido el presupuesto de las cuatro
#: combinaciones en una sola.
REJILLA = [{"oculta": o, "lr": l} for o in (16, 32) for l in (0.01, 0.03)]

#: Se mide validación cada 50 épocas y se para si no mejora en 300. El sondeo
#: previo mostró que la convergencia es lenta pero monótona: a 180 épocas la
#: curva sigue subiendo con las tres tasas probadas, así que el presupuesto se
#: fija generoso y se deja que la parada temprana decida.
CADA = 50
PACIENCIA = 300

#: El ajuste largo puede permitirse más paciencia: ya no compite con otras
#: configuraciones por el presupuesto.
PACIENCIA_LARGA = 600


def entrenar(modo, epocas, segundos, abanico):
    """Recorre la rejilla con punto de control por configuración.

    El entorno mata los procesos largos, así que el estado se persiste tras
    cada bloque de épocas y la ejecución siguiente reanuda donde quedó. Es el
    mismo patrón que `rejilla_incremental` de `10_fuga_variantes.py`.
    """
    os.makedirs(SALIDA, exist_ok=True)
    df, X, y, i_aj, i_val, i_te, B, Pp, Pe = preparar()
    peso_pos = float((y[i_aj] == 0).sum() / max((y[i_aj] == 1).sum(), 1))
    est = ruta_estado(modo)

    #: Estado global de la rejilla: qué configuraciones están cerradas y con
    #: qué resultado de validación.
    ruta_rej = os.path.join(SALIDA, f"rejilla_{modo}.csv")
    hechas = {}
    if os.path.exists(ruta_rej):
        for r in pd.read_csv(ruta_rej).to_dict("records"):
            hechas[(r["oculta"], r["lr"])] = r

    for cfg in REJILLA:
        clave = (cfg["oculta"], cfg["lr"])
        if clave in hechas:
            continue
        rng = np.random.default_rng(SEMILLA)
        red = RedDeGrafos(X.shape[1], cfg["oculta"], modo, rng)
        muestras = (juego_de_muestras(B, abanico) if modo == "sage" else None)
        m = {k: np.zeros_like(v) for k, v in red.parametros().items()}
        v = {k: np.zeros_like(x) for k, x in red.parametros().items()}
        e, mejor_val, mejor_e = 0, -1.0, 0
        mejor_par = {k: x.copy() for k, x in red.parametros().items()}
        historia = []

        marca = os.path.join(SALIDA, f"_ck_{modo}_{cfg['oculta']}_{cfg['lr']}.npz")
        if os.path.exists(marca):
            d = np.load(marca, allow_pickle=True)
            red.cargar_parametros({k: d[f"p_{k}"] for k in red.nombres()})
            m = {k: d[f"m_{k}"] for k in red.nombres()}
            v = {k: d[f"v_{k}"] for k in red.nombres()}
            mejor_par = {k: d[f"b_{k}"] for k in red.nombres()}
            e = int(d["epoca"]); mejor_val = float(d["mejor_val"])
            mejor_e = int(d["mejor_e"]); historia = list(d["historia"])
            print(f"[{modo} h{cfg['oculta']} lr{cfg['lr']}] reanudado en {e}",
                  flush=True)

        b1, b2, eps = 0.9, 0.999, 1e-8
        t0 = time.time()
        while e < epocas and time.time() - t0 < segundos:
            e += 1
            Ppe, Pee = (muestras[e % len(muestras)] if modo == "sage"
                        else (Pp, Pe))

            z, c = red.adelante(X, Ppe, Pee)
            dz = np.zeros(len(y))
            dz[i_aj] = d_perdida(z[i_aj], y[i_aj], peso_pos)
            g = red.atras(c, dz, Ppe, Pee)
            for k in red.nombres():
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** e)
                vh = v[k] / (1 - b2 ** e)
                setattr(red, k, getattr(red, k) - cfg["lr"] * mh / (np.sqrt(vh) + eps))

            if e % CADA == 0:
                zc, _ = red.adelante(X, Pp, Pe, guardar=False)
                av = roc_auc_score(y[i_val], zc[i_val])
                historia.append({"epoca": e,
                                 "perdida": round(float(perdida(z[i_aj], y[i_aj], peso_pos)), 6),
                                 "auc_validacion": round(float(av), 6)})
                if av > mejor_val:
                    mejor_val, mejor_e = float(av), e
                    mejor_par = {k: x.copy() for k, x in red.parametros().items()}
                if e - mejor_e >= PACIENCIA:
                    print(f"[{modo} h{cfg['oculta']} lr{cfg['lr']}] "
                          f"parada temprana en {e}", flush=True)
                    e = epocas
                    break

        np.savez(marca, epoca=e, mejor_val=mejor_val, mejor_e=mejor_e,
                 historia=np.array(historia, dtype=object),
                 **{f"p_{k}": getattr(red, k) for k in red.nombres()},
                 **{f"m_{k}": m[k] for k in red.nombres()},
                 **{f"v_{k}": v[k] for k in red.nombres()},
                 **{f"b_{k}": mejor_par[k] for k in red.nombres()})

        if e < epocas:
            ult = historia[-1] if historia else {"epoca": e, "auc_validacion": float("nan")}
            print(f"[{modo} h{cfg['oculta']} lr{cfg['lr']}] punto de control en "
                  f"{e}/{epocas} · validación {ult['auc_validacion']:.4f}",
                  flush=True)
            return False

        #: Configuración cerrada: se guarda su mejor validación y se sigue.
        hechas[clave] = {"oculta": cfg["oculta"], "lr": cfg["lr"],
                         "mejor_epoca": mejor_e,
                         "auc_validacion": round(mejor_val, 6)}
        pd.DataFrame(list(hechas.values())).to_csv(ruta_rej, index=False)
        pd.DataFrame(historia).to_csv(
            os.path.join(SALIDA, f"historia_{modo}_{cfg['oculta']}_{cfg['lr']}.csv"),
            index=False)
        print(f"[{modo} h{cfg['oculta']} lr{cfg['lr']}] cerrada · "
              f"validación {mejor_val:.4f} en la época {mejor_e}", flush=True)

    #: Toda la rejilla cerrada: se recupera la mejor configuración y se evalúa
    #: **una sola vez** sobre el conjunto de prueba.
    rej = pd.DataFrame(list(hechas.values())).sort_values(
        "auc_validacion", ascending=False)
    mejor = rej.iloc[0]
    marca = os.path.join(SALIDA, f"_ck_{modo}_{int(mejor['oculta'])}_{mejor['lr']}.npz")
    d = np.load(marca, allow_pickle=True)
    red = RedDeGrafos(X.shape[1], int(mejor["oculta"]), modo,
                      np.random.default_rng(SEMILLA))
    red.cargar_parametros({k: d[f"b_{k}"] for k in red.nombres()})

    #: La evaluación usa el vecindario completo también en GraphSAGE: el
    #: muestreo es una técnica de entrenamiento y no debe meter azar en la
    #: cifra publicada.
    z, _ = red.adelante(X, Pp, Pe, guardar=False)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    pd.DataFrame({"id_nodo": df["id_nodo"].values[i_te],
                  "y": y[i_te], "p": p[i_te]}).to_csv(
        os.path.join(SALIDA, f"_pred_{modo}.csv"), index=False)
    met = metricas(y[i_te], p[i_te])
    with open(os.path.join(SALIDA, f"metricas_{modo}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"modelo": ETIQUETAS[modo],
                   "oculta": int(mejor["oculta"]), "lr": float(mejor["lr"]),
                   "mejor_epoca": int(mejor["mejor_epoca"]),
                   "auc_validacion": float(mejor["auc_validacion"]),
                   "combinaciones_exploradas": len(REJILLA),
                   "abanico": abanico if modo == "sage" else None,
                   **{k: round(float(x), 4) for k, x in met.items()}},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n[{modo}] TERMINADO · h{int(mejor['oculta'])} lr{mejor['lr']} "
          f"época {int(mejor['mejor_epoca'])}")
    for k, x in met.items():
        print(f"  {k:<14} {x:.4f}")
    return True


def ajuste_largo(modo, epocas, segundos, abanico):
    """Reajusta la configuración ganadora con presupuesto amplio.

    La rejilla se resuelve con 1.200 épocas por combinación, que basta para
    ordenarlas entre sí pero no para agotar ninguna: las dos ganadoras seguían
    mejorando en validación al llegar al tope. Publicar ahí sería medir el
    presupuesto y no la arquitectura, que es justo lo que el apartado 5.2.4
    reprocha a la versión anterior del trabajo.

    Se retoma la ganadora desde su punto de control —con el estado de Adam
    incluido, para que la continuación sea indistinguible de un entrenamiento
    seguido— y se deja correr hasta que la parada temprana decida.
    """
    os.makedirs(SALIDA, exist_ok=True)
    df, X, y, i_aj, i_val, i_te, B, Pp, Pe = preparar()
    peso_pos = float((y[i_aj] == 0).sum() / max((y[i_aj] == 1).sum(), 1))
    rej = pd.read_csv(os.path.join(SALIDA, f"rejilla_{modo}.csv")).sort_values(
        "auc_validacion", ascending=False)
    cfg = rej.iloc[0]
    oculta, lr = int(cfg["oculta"]), float(cfg["lr"])

    largo = os.path.join(SALIDA, f"_largo_{modo}.npz")
    origen = (largo if os.path.exists(largo)
              else os.path.join(SALIDA, f"_ck_{modo}_{oculta}_{lr}.npz"))
    d = np.load(origen, allow_pickle=True)
    red = RedDeGrafos(X.shape[1], oculta, modo, np.random.default_rng(SEMILLA))
    red.cargar_parametros({k: d[f"p_{k}"] for k in red.nombres()})
    m = {k: d[f"m_{k}"] for k in red.nombres()}
    v = {k: d[f"v_{k}"] for k in red.nombres()}
    mejor_par = {k: d[f"b_{k}"] for k in red.nombres()}
    e = int(d["epoca"]); mejor_val = float(d["mejor_val"])
    mejor_e = int(d["mejor_e"]); historia = list(d["historia"])
    muestras = juego_de_muestras(B, abanico) if modo == "sage" else None
    print(f"[{modo} largo] h{oculta} lr{lr} desde la época {e}, "
          f"validación {mejor_val:.4f}", flush=True)

    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    parada = False
    while e < epocas and time.time() - t0 < segundos:
        e += 1
        Ppe, Pee = (muestras[e % len(muestras)] if modo == "sage" else (Pp, Pe))
        z, c = red.adelante(X, Ppe, Pee)
        dz = np.zeros(len(y), dtype=np.float32)
        dz[i_aj] = d_perdida(z[i_aj], y[i_aj], peso_pos)
        g = red.atras(c, dz, Ppe, Pee)
        for k in red.nombres():
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            setattr(red, k, getattr(red, k)
                    - lr * (m[k] / (1 - b1 ** e)) / (np.sqrt(v[k] / (1 - b2 ** e)) + eps))
        if e % CADA == 0:
            zc, _ = red.adelante(X, Pp, Pe, guardar=False)
            av = roc_auc_score(y[i_val], zc[i_val])
            historia.append({"epoca": e,
                             "perdida": round(float(perdida(z[i_aj], y[i_aj], peso_pos)), 6),
                             "auc_validacion": round(float(av), 6)})
            if av > mejor_val:
                mejor_val, mejor_e = float(av), e
                mejor_par = {k: x.copy() for k, x in red.parametros().items()}
            if e - mejor_e >= PACIENCIA_LARGA:
                print(f"[{modo} largo] parada temprana en {e}", flush=True)
                parada = True
                break

    np.savez(largo, epoca=e, mejor_val=mejor_val, mejor_e=mejor_e,
             historia=np.array(historia, dtype=object),
             **{f"p_{k}": getattr(red, k) for k in red.nombres()},
             **{f"m_{k}": m[k] for k in red.nombres()},
             **{f"v_{k}": v[k] for k in red.nombres()},
             **{f"b_{k}": mejor_par[k] for k in red.nombres()})

    if not parada and e < epocas:
        print(f"[{modo} largo] punto de control en {e}/{epocas} · "
              f"mejor validación {mejor_val:.4f} en {mejor_e}", flush=True)
        return False

    red.cargar_parametros(mejor_par)
    z, _ = red.adelante(X, Pp, Pe, guardar=False)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    pd.DataFrame({"id_nodo": df["id_nodo"].values[i_te],
                  "y": y[i_te], "p": p[i_te]}).to_csv(
        os.path.join(SALIDA, f"_pred_{modo}.csv"), index=False)
    pd.DataFrame(historia).to_csv(
        os.path.join(SALIDA, f"historia_{modo}.csv"), index=False)
    met = metricas(y[i_te], p[i_te])
    with open(os.path.join(SALIDA, f"metricas_{modo}.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"modelo": ETIQUETAS[modo], "oculta": oculta, "lr": lr,
                   "epocas_recorridas": int(e), "mejor_epoca": int(mejor_e),
                   "auc_validacion": round(float(mejor_val), 6),
                   "combinaciones_exploradas": len(REJILLA),
                   "abanico": abanico if modo == "sage" else None,
                   **{k: round(float(x), 4) for k, x in met.items()}},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n[{modo}] AJUSTE LARGO TERMINADO · época {mejor_e} de {e}")
    for k, x in met.items():
        print(f"  {k:<14} {x:.4f}")
    return True


def comprobar_gradiente(modo, oculta=6, n=400, tol=1e-5):
    """Contrasta el gradiente analítico contra diferencias finitas centradas.

    Es la única garantía de que la retropropagación escrita a mano es
    correcta. Se hace sobre un subgrafo pequeño y en doble precisión, porque
    en `float32` el error de redondeo de la diferencia finita domina y la
    comprobación no distingue un fallo real de ruido numérico.
    """
    df, X, y, i_aj, i_val, i_te, B, Pp, Pe = preparar()
    rng = np.random.default_rng(SEMILLA)
    sub = rng.choice(len(y), size=n, replace=False)
    Bs = B[sub][:, np.asarray(B[sub].sum(0)).ravel() > 0]
    gp = np.asarray(Bs.sum(1)).ravel()
    ge = np.asarray(Bs.sum(0)).ravel()
    gp[gp == 0] = 1.0
    ge[ge == 0] = 1.0
    Pps = (sparse.diags(1.0 / gp) @ Bs).tocsr()
    Pes = (sparse.diags(1.0 / ge) @ Bs.T).tocsr()
    Xs, ys = X[sub], y[sub]

    red = RedDeGrafos(Xs.shape[1], oculta, modo, rng, dtype=np.float64)
    peso_pos = float((ys == 0).sum() / max((ys == 1).sum(), 1))
    #: Los sesgos arrancan en cero y en ese punto su gradiente es exactamente
    #: nulo: la reponderación de clase hace que los residuos de positivos y
    #: negativos se cancelen. Comprobar ahí no distingue una derivada correcta
    #: de una que devuelve cero siempre, así que se evalúa en un punto
    #: cualquiera del espacio de parámetros.
    for k in ("b1", "b2", "b3"):
        setattr(red, k, rng.normal(scale=0.3, size=getattr(red, k).shape))

    z, c = red.adelante(Xs, Pps, Pes)
    dz = d_perdida(z, ys, peso_pos)
    g = red.atras(c, dz, Pps, Pes)

    print(f"\ncomprobación de gradiente — {ETIQUETAS[modo]}")
    peor_global = 0.0
    for k in red.nombres():
        P = getattr(red, k)
        idxs = [tuple(rng.integers(0, s) for s in P.shape) for _ in range(6)]
        peor = 0.0
        for ix in idxs:
            orig = P[ix]
            h = 1e-6 * max(1.0, abs(orig))
            P[ix] = orig + h
            zp, _ = red.adelante(Xs, Pps, Pes, guardar=False)
            lp = perdida(zp, ys, peso_pos)
            P[ix] = orig - h
            zm, _ = red.adelante(Xs, Pps, Pes, guardar=False)
            lm = perdida(zm, ys, peso_pos)
            P[ix] = orig
            num = (lp - lm) / (2 * h)
            ana = float(g[k][ix])
            #: Criterio mixto: relativo cuando la derivada tiene magnitud, y
            #: absoluto cuando es próxima a cero. Un cociente puro sobre una
            #: derivada nula da 1,0 aunque las dos estimaciones coincidan en
            #: el ruido de redondeo, y eso es un falso positivo, no un fallo.
            if abs(num - ana) < 1e-9:
                rel = 0.0
            else:
                rel = abs(num - ana) / max(abs(num), abs(ana), 1e-9)
            peor = max(peor, rel)
        peor_global = max(peor_global, peor)
        print(f"  {k:<4} error relativo {peor:.2e}")
    ok = peor_global < tol
    print(f"\n{'CORRECTO' if ok else 'FALLA'}: error relativo peor "
          f"{peor_global:.2e} (tolerancia {tol:.0e})")
    return ok


def fase_tabla():
    """Amplía la tabla de ordenamientos con las dos arquitecturas nuevas."""
    base = pd.read_csv(rutas.artefacto("tabla_5_21_siete_ordenamientos.csv"))
    filas = []
    for modo in ("gcn", "sage"):
        #: Con `artefacto` y no contra `salidas/` a secas: la tabla se arma
        #: igual desde el entrenamiento propio que desde el paquete de datos,
        #: que es lo que permite rehacerla sin repetir horas de cómputo.
        ruta = rutas.artefacto(os.path.join("gnn", f"metricas_{modo}.json"))
        if not os.path.exists(ruta):
            print(f"falta {ruta}; entrena primero {modo}")
            continue
        d = json.load(open(ruta, encoding="utf-8"))
        filas.append({"Ordenamiento": d["modelo"], "Tipo": "entrenado",
                      "AUC-ROC": d["AUC-ROC"], "PR-AUC": d["PR-AUC"],
                      "F1": d["F1"], "Precision@200": d["Precision@200"],
                      "Recall@200": d["Recall@200"]})
    if not filas:
        return
    t = pd.concat([base, pd.DataFrame(filas)], ignore_index=True)
    destino = os.path.join(MODELOS, "tabla_5_29_nueve_ordenamientos.csv")
    t.to_csv(destino, index=False)
    print(t.to_string(index=False))
    print(f"\n-> {destino}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", required=True,
                    choices=["gradiente", "entrenar", "largo", "tabla"])
    ap.add_argument("--modelo", choices=["gcn", "sage"], default="gcn")
    ap.add_argument("--epocas", type=int, default=1200)
    ap.add_argument("--abanico", type=int, default=25)
    ap.add_argument("--segundos", type=int, default=30)
    a = ap.parse_args()

    if a.fase == "gradiente":
        ok = all(comprobar_gradiente(m) for m in ("gcn", "sage"))
        raise SystemExit(0 if ok else 1)
    if a.fase == "tabla":
        fase_tabla()
        return
    if a.fase == "largo":
        ajuste_largo(a.modelo, max(a.epocas, 4000), a.segundos, a.abanico)
        return
    entrenar(a.modelo, a.epocas, a.segundos, a.abanico)


if __name__ == "__main__":
    main()
