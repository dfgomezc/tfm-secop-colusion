#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicación de las alertas de mayor puntuación (OE-6).

Responde al sexto objetivo específico: producir, sobre las alertas de mayor
puntuación, explicaciones que un auditor pueda verificar contra el expediente.

El objeto de la explicación es el **modelo relacional del trabajo** —la
propagación de características al estilo SGC seguida de un perceptrón
multicapa—, que es el que no admite lectura directa: sus pesos no dicen nada
interpretable y su salida es una probabilidad sin justificación. Un conjunto de
árboles no plantea ese problema, porque expone importancia de variables y reglas
inspeccionables; por eso el modelo neuronal es el caso que hay que abrir, y el
Random Forest sirve aquí de referencia con la que contrastar lo que se obtenga.

## Qué produce

Tres niveles de atribución sobre el mismo modelo y la misma partición con que
se obtienen los resultados del Capítulo 5, encadenados de lo general a lo
concreto:

1. **Por variable.** Valores de Shapley (Lundberg y Lee, 2017). Para el modelo
   neuronal se estiman con el método por núcleos, que no supone nada sobre la
   arquitectura y por eso sirve para una caja negra; para el conjunto de
   árboles se calculan con el algoritmo exacto de Lundberg et al. (2020). En
   ambos casos la suma de las atribuciones más el valor base reconstruye la
   probabilidad predicha, y esa identidad se comprueba.

2. **Por bloque de propagación.** Las veintiuna columnas de entrada son las
   siete variables del nodo, su media a un salto y su media a dos saltos. Sumar
   las atribuciones dentro de cada bloque responde a la pregunta que el trabajo
   plantea desde el Capítulo 1: qué parte del riesgo que se imputa a un actor
   procede de él y qué parte de su entorno.

3. **Por vecino.** La propagación es una combinación lineal de las variables de
   los vecinos, de modo que la atribución de una columna propagada se reparte
   entre ellos sin aproximación: si la variable propagada $j$ del nodo $i$ vale
   $\\hat{A}_{i\\cdot} X_{\\cdot j}$, la parte del vecino $k$ es
   $\\hat{A}_{ik} X_{kj}$ dividida por ese total. El resultado es el subgrafo
   responsable de la alerta: qué actores concretos la sostienen y en qué
   proporción.

El tercer nivel es el que convierte una puntuación en un argumento verificable:
nombra a los actores del entorno que la explican, y sobre ellos se puede pedir
el expediente.

## Fidelidad

La explicación no vale nada si no es del modelo que se publicó. El módulo
reajusta el modelo con la configuración ganadora de su rejilla y comprueba que
su ordenamiento es el publicado —AUC-ROC y precisión sobre la lista corta,
dentro de la tolerancia declarada más abajo— antes de explicar nada.

Comprueba además la identidad que hace verificable a la propia explicación: la
suma de las atribuciones más el valor base reconstruye la puntuación del
modelo. Si esa suma no cierra, la explicación no describe lo que el modelo
hace.

## Uso

    python -m tfm.modelos.explicabilidad                  # el modelo neuronal
    python -m tfm.modelos.explicabilidad --modelo rf      # la referencia
    python -m tfm.modelos.explicabilidad --comparar       # las dos, y su cotejo
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from tfm import rutas
from tfm.modelos import fuga_variantes as fuga

SEMILLA = 42
SALIDA = os.path.join(rutas.resultados(), "explicabilidad")

#: Las siete variables que sobreviven al control de fuga del apartado 5.2.2.
#: Se explican las mismas con las que se entrena: explicar sobre otro conjunto
#: describiría un modelo que no es el que produce las alertas.
FEATURES = fuga.FEATURES_V1

BLOQUES = ("propio", "un salto", "dos saltos")

#: Los dos modelos que se pueden explicar, con la rejilla de la que se lee su
#: configuración ganadora y la predicción publicada contra la que se comprueba
#: que el reajuste es el mismo modelo. El neuronal se ajusta sobre las
#: variables propagadas por definición; el bosque, sobre las mismas, que es la
#: configuración con la que gana la comparativa del apartado 5.2.3.
MODELOS = {
    "mlp": {
        "etiqueta": "SGC — propagación + MLP",
        "rejilla": os.path.join("v1", "rejilla_gnn.csv"),
        "publicado": os.path.join("v1", "_pred_gnn.csv"),
    },
    "rf": {
        "etiqueta": "ML — Random Forest sobre variables propagadas",
        "rejilla": os.path.join("matriz", "propagadas", "_parcial_rf.csv"),
        "publicado": os.path.join("matriz", "propagadas", "_pred_rf.csv"),
    },
}

#: Nombres legibles de las variables, para las tablas publicadas.
LEGIBLE = {
    "strength": "grado ponderado",
    "pagerank": "PageRank",
    "en_componente_gigante": "en componente gigante",
    "log_valor_total": "log del valor total",
    "log_valor_medio": "log del valor medio",
    "n_procesos": "número de procesos",
    "n_cuentas_plataforma": "cuentas en plataforma",
}

#: Tolerancia con que se acepta que el modelo reajustado es el publicado. Se
#: fija sobre las métricas y no sobre las probabilidades una a una porque el
#: árbol resuelve los empates de corte en función del orden en que la
#: biblioteca recorre los candidatos, y ese orden cambia entre versiones de
#: NumPy: dos entornos distintos ajustan el mismo bosque y difieren en la
#: cuarta cifra de alguna probabilidad. Lo que no puede moverse es el
#: ordenamiento que se publica, y eso es lo que se comprueba.
TOLERANCIA_AUC = 1e-3
TOLERANCIA_P200 = 0.01

#: Tamaño del fondo de referencia y número de coaliciones del método por
#: núcleos. Con veintiuna variables, las 2²¹ coaliciones exactas no son
#: abordables; 512 muestras dejan el residuo de reconstrucción en el orden de
#: 10⁻¹⁶, que es el de la aritmética de coma flotante.
FONDO = 60
COALICIONES = 512


# ---------------------------------------------------------------------------
def dir_modelo(modelo):
    return os.path.join(SALIDA, modelo)


def mejor_configuracion(modelo):
    """La combinación ganadora de la rejilla publicada, tal como se guardó.

    Las dos rejillas no tienen el mismo formato: la del bosque viene del
    recorrido incremental y guarda la combinación entera en una columna
    `combo`; la del perceptrón viene de la comparativa y guarda una columna por
    hiperparámetro. Se admiten las dos formas.
    """
    rej = pd.read_csv(rutas.artefacto(MODELOS[modelo]["rejilla"]))
    fila = rej.sort_values("CV AUC", ascending=False).iloc[0]
    if "combo" in rej.columns:
        cfg = dict(eval(fila["combo"]))
    else:
        cfg = {c: fila[c] for c in rej.columns if c not in ("CV AUC", "sigma")}
        #: `hidden_layer_sizes` se guarda como el texto de la tupla.
        for k, v in list(cfg.items()):
            if isinstance(v, str) and v.strip().startswith("("):
                cfg[k] = eval(v)
            elif isinstance(v, float) and v.is_integer():
                cfg[k] = int(v)
    print(f"configuración ganadora  {cfg}  ·  CV AUC {fila['CV AUC']:.4f}")
    return cfg


def operadores(ids, aristas):
    """El operador de propagación normalizado, además de la matriz propagada.

    `fuga_variantes.vecindario()` devuelve solo el resultado, que es lo que
    necesita el modelo. Aquí hace falta también el operador, porque el reparto
    por vecino se hace sobre sus coeficientes. Se construye igual y se
    comprueba después que ambas vías coinciden.
    """
    idx = {n: i for i, n in enumerate(ids)}
    ent, filas, cols = {}, [], []
    for e, n in aristas:
        if n not in idx:
            continue
        filas.append(idx[n])
        cols.append(ent.setdefault(e, len(ent)))
    B = sparse.csr_matrix((np.ones(len(filas)), (filas, cols)),
                          shape=(len(ids), len(ent)))
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    g = np.asarray(A.sum(axis=1)).ravel()
    g[g == 0] = 1
    return (sparse.diags(1.0 / g) @ A).tocsr()


def preparar():
    """Datos, representación y partición, idénticos a los del apartado 5.2.3."""
    df, aristas = fuga.cargar()
    y = df["riesgo_alto"].astype(int).values
    Xn = StandardScaler().fit_transform(df[FEATURES].astype(float).values)
    X = fuga.vecindario(df["id_nodo"].tolist(), Xn, aristas)

    Ah = operadores(df["id_nodo"].tolist(), aristas)
    h1 = np.asarray(Ah @ Xn)
    h2 = np.asarray(Ah @ h1)
    #: La comprobación que autoriza a usar el operador para repartir: si la
    #: reconstrucción no coincide con la del pipeline, el reparto por vecino
    #: estaría describiendo otra propagación.
    if not np.allclose(X, np.hstack([Xn, h1, h2]), atol=1e-9):
        sys.exit("el operador reconstruido no coincide con la propagación del "
                 "pipeline; no se puede repartir por vecino")

    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.30,
                                  random_state=SEMILLA, stratify=y)
    return df, X, Xn, h1, Ah, y, i_tr, i_te


def ajustar(modelo, X, y, i_tr, i_te, cfg):
    """Reajusta el modelo publicado y comprueba que lo es."""
    from sklearn.metrics import roc_auc_score

    if modelo == "mlp":
        m = MLPClassifier(max_iter=250, early_stopping=True,
                          random_state=SEMILLA, **cfg)
    else:
        m = RandomForestClassifier(random_state=SEMILLA, n_jobs=2, **cfg)
    m.fit(X[i_tr], y[i_tr])
    p = m.predict_proba(X[i_te])[:, 1]
    auc = float(roc_auc_score(y[i_te], p))
    p200 = float(y[i_te][np.argsort(-p)[:200]].mean())

    control = {"AUC-ROC": round(auc, 6), "Precision@200": round(p200, 6)}
    publicado = rutas.artefacto(MODELOS[modelo]["publicado"])
    if os.path.exists(publicado):
        pub = pd.read_csv(publicado)
        auc_pub = float(roc_auc_score(pub["y"], pub["p"]))
        p200_pub = float(pub["y"].values[np.argsort(-pub["p"].values)[:200]].mean())
        control.update({"AUC-ROC publicada": round(auc_pub, 6),
                        "Precision@200 publicada": round(p200_pub, 6),
                        "coincidencia top-200": int(len(
                            set(np.argsort(-p)[:200])
                            & set(np.argsort(-pub["p"].values)[:200])))})
        print(f"AUC {auc:.6f} frente a {auc_pub:.6f} publicada · "
              f"Precision@200 {p200:.3f} frente a {p200_pub:.3f}")
        if (abs(auc - auc_pub) > TOLERANCIA_AUC
                or abs(p200 - p200_pub) > TOLERANCIA_P200):
            sys.exit("el modelo reajustado no reproduce el publicado; "
                     "la explicación no describiría el modelo del capítulo 5")
    return m, p, control


def atribuir(modelo, m, X, i_tr, i_te, orden):
    """Valores de Shapley del modelo, sobre las alertas seleccionadas.

    Para el bosque se emplea el algoritmo exacto para conjuntos de árboles.
    Para el perceptrón no hay algoritmo exacto abordable —veintiuna variables
    son 2²¹ coaliciones— y se estima por el método de núcleos, que trata al
    modelo como una caja negra: solo lo consulta, no mira dentro. Es lo que
    permite aplicarlo a una red neuronal sin suponer nada de su arquitectura.
    """
    import shap

    if modelo == "rf":
        ex = shap.TreeExplainer(m, model_output="raw")
        bruto = ex.shap_values(X[i_te][orden])
        if isinstance(bruto, list):
            phi, base = bruto[1], float(np.ravel(ex.expected_value)[1])
        elif bruto.ndim == 3:
            phi, base = bruto[:, :, 1], float(np.ravel(ex.expected_value)[1])
        else:
            phi, base = bruto, float(np.ravel(ex.expected_value)[0])
        return phi, base

    #: El fondo de referencia es lo que el método por núcleos toma como
    #: «ausencia» de una variable. Se muestrea con semilla para que dos
    #: ejecuciones den la misma explicación.
    fondo = shap.sample(X[i_tr], FONDO, random_state=SEMILLA)
    ex = shap.KernelExplainer(lambda z: m.predict_proba(z)[:, 1], fondo)
    phi = ex.shap_values(X[i_te][orden], nsamples=COALICIONES, silent=True)
    return np.asarray(phi), float(np.ravel(ex.expected_value)[0])


def repartir_por_vecino(i, phi_fila, Ah, Xn, h1, d):
    """Reparte entre los vecinos la atribución de las columnas propagadas.

    La propagación es lineal, de modo que el reparto es exacto y no una
    heurística: la parte del vecino `k` en la variable propagada `j` es su
    aportación a la media, `Â_ik · X_kj`, sobre el total `h_ij`. Solo se
    reparten las columnas cuyo total no sea numéricamente cero, porque ahí el
    cociente no significa nada.
    """
    fila1 = Ah[i]
    #: El vecindario a dos saltos se obtiene componiendo la fila con el
    #: operador, sin materializar Â² para los 82.984 nodos.
    fila2 = (fila1 @ Ah).tocsr()

    reparto = defaultdict(float)
    for bloque, fila, base_col, valores in (
            ("un salto", fila1, d, Xn), ("dos saltos", fila2, 2 * d, h1)):
        idx, pesos = fila.indices, fila.data
        if len(idx) == 0:
            continue
        for j in range(d):
            total = float(pesos @ valores[idx, j])
            if abs(total) < 1e-12:
                continue
            aporte = phi_fila[base_col + j] * (pesos * valores[idx, j]) / total
            for k, a in zip(idx, aporte):
                reparto[(bloque, int(k))] += float(a)
    return reparto


# ---------------------------------------------------------------------------
def explicar(modelo, cuantas, vecinos, datos=None):
    destino = dir_modelo(modelo)
    os.makedirs(destino, exist_ok=True)
    print(f"\n=== {MODELOS[modelo]['etiqueta']}")
    cfg = mejor_configuracion(modelo)
    df, X, Xn, h1, Ah, y, i_tr, i_te = datos if datos else preparar()
    d = len(FEATURES)
    m, p, control = ajustar(modelo, X, y, i_tr, i_te, cfg)

    orden = np.argsort(-p)[:cuantas]
    globales = i_te[orden]
    phi, base = atribuir(modelo, m, X, i_tr, i_te, orden)

    #: La identidad que hace verificable la explicación: lo atribuido más el
    #: valor base reconstruye la puntuación.
    error = float(np.max(np.abs(base + phi.sum(axis=1) - p[orden])))
    print(f"error máximo de reconstrucción  {error:.2e}")
    if error > 1e-6:
        sys.exit("las atribuciones no reconstruyen la puntuación")

    seudo = pd.read_parquet(rutas.parquet_de("nodo_seudonimo",
                                             "nodo_seudonimo.parquet"))
    seudo = seudo.set_index("id_nodo")["seudonimo"].to_dict()
    ids = df["id_nodo"].values

    def alias(k):
        return seudo.get(ids[k], "sin seudónimo")

    # -- nivel 1: por variable ---------------------------------------------
    columnas = [(b, f) for b in BLOQUES for f in FEATURES]
    filas = []
    for r, k in enumerate(globales):
        for c, (bloque, var) in enumerate(columnas):
            filas.append({"id_nodo": ids[k], "seudonimo": alias(k),
                          "bloque": bloque, "variable": var,
                          "atribucion": round(float(phi[r, c]), 6)})
    pd.DataFrame(filas).to_csv(
        os.path.join(destino, "atribucion_variables.csv"), index=False)

    # -- nivel 2: por bloque -----------------------------------------------
    bloq = pd.DataFrame({
        "id_nodo": ids[globales],
        "seudonimo": [alias(k) for k in globales],
        "puntuacion": np.round(p[orden], 6),
        "valor_base": np.round(np.full(len(globales), base), 6),
        "propio": np.round(phi[:, :d].sum(1), 6),
        "un_salto": np.round(phi[:, d:2 * d].sum(1), 6),
        "dos_saltos": np.round(phi[:, 2 * d:].sum(1), 6),
        "etiqueta": y[globales],
    })
    bloq["entorno"] = np.round(bloq["un_salto"] + bloq["dos_saltos"], 6)
    bloq.to_csv(os.path.join(destino, "atribucion_bloques.csv"), index=False)

    # -- nivel 3: por vecino -----------------------------------------------
    filas = []
    for r, k in enumerate(globales):
        reparto = repartir_por_vecino(k, phi[r], Ah, Xn, h1, d)
        mejores = sorted(reparto.items(), key=lambda t: -abs(t[1]))[:vecinos]
        for rango, ((bloque, vecino), valor) in enumerate(mejores, 1):
            filas.append({"id_nodo": ids[k], "seudonimo": alias(k),
                          "rango": rango, "bloque": bloque,
                          "vecino": ids[vecino],
                          "vecino_seudonimo": alias(vecino),
                          "atribucion": round(float(valor), 6)})
    v3 = pd.DataFrame(filas)
    v3.to_csv(os.path.join(destino, "subgrafo_responsable.csv"), index=False)

    # -- resumen y tabla publicable ----------------------------------------
    ban = pd.read_csv(rutas.artefacto("banderas_rojas.csv")).set_index("id_nodo")
    nombres_ban = ["en_triangulo", "comparte_contacto",
                   "representante_compartido", "componente_cerrado"]
    presentes = [n for n in ids[globales] if n in ban.index]
    con_bandera = int((ban.loc[presentes, nombres_ban].sum(axis=1) > 0).sum())

    tabla = []
    for r, k in enumerate(globales[:10]):
        c = int(np.argmax(np.abs(phi[r])))
        bloque_var, var = columnas[c]
        vecinos_k = v3[v3["id_nodo"] == ids[k]].head(1)
        total = float(np.abs(phi[r]).sum())
        tabla.append({
            "Actor": alias(k),
            "Puntuación": round(float(p[orden][r]), 4),
            "Propio %": round(100 * np.abs(phi[r, :d]).sum() / total, 1),
            "Entorno %": round(100 * np.abs(phi[r, d:]).sum() / total, 1),
            "Variable dominante": f"{LEGIBLE.get(var, var)} ({bloque_var})",
            "Vecino que más aporta": (vecinos_k["vecino_seudonimo"].iloc[0]
                                      if len(vecinos_k) else "—"),
            "Aportación de ese vecino": (
                round(float(vecinos_k["atribucion"].iloc[0]), 4)
                if len(vecinos_k) else float("nan")),
        })
    if modelo == "mlp":
        pd.DataFrame(tabla).to_csv(
            os.path.join(rutas.resultados(),
                         "tabla_5_30_explicabilidad.csv"), index=False)

    resumen = {
        "modelo": MODELOS[modelo]["etiqueta"],
        "metodo": ("Shapley por núcleos, tratando al modelo como caja negra"
                   if modelo == "mlp" else
                   "Shapley exacto para conjuntos de árboles"),
        "alertas_explicadas": int(cuantas),
        "configuracion": {k: str(v) for k, v in cfg.items()},
        "valor_base": round(base, 6),
        "control_del_modelo": control,
        "error_maximo_reconstruccion": error,
        "reparto_medio_propio": round(float(
            (np.abs(phi[:, :d]).sum(1) / np.abs(phi).sum(1)).mean() * 100), 2),
        "reparto_medio_entorno": round(float(
            (np.abs(phi[:, d:]).sum(1) / np.abs(phi).sum(1)).mean() * 100), 2),
        "alertas_con_entorno_dominante": int(
            (np.abs(phi[:, d:]).sum(1) > np.abs(phi[:, :d]).sum(1)).sum()),
        "positivos_en_el_top": int(y[globales].sum()),
        "alertas_con_bandera_roja": con_bandera,
        "vecinos_distintos_en_los_subgrafos": int(v3["vecino"].nunique()),
    }
    with open(os.path.join(destino, "resumen.json"), "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"-> {destino}")
    return (df, X, Xn, h1, Ah, y, i_tr, i_te)


def comparar():
    """Coteja la explicación de la caja negra con la del modelo interpretable.

    Es la comprobación que decide si abrir el modelo neuronal aporta algo: si
    las dos explicaciones coincidieran, bastaría con leer el bosque.
    """
    filas = []
    lecturas = {}
    for modelo in ("mlp", "rf"):
        ruta = os.path.join(dir_modelo(modelo), "atribucion_variables.csv")
        if not os.path.exists(ruta):
            sys.exit(f"falta {ruta}; ejecuta antes --modelo {modelo}")
        lecturas[modelo] = pd.read_csv(ruta)

    a, b = lecturas["mlp"], lecturas["rf"]
    clave = ["id_nodo", "bloque", "variable"]
    j = a.merge(b, on=clave, suffixes=("_mlp", "_rf"))
    corr = float(np.corrcoef(j["atribucion_mlp"], j["atribucion_rf"])[0, 1])

    #: Cuántas alertas comparten las dos explicaciones. Se comparan por el
    #: actor señalado, no por la cifra: dos modelos distintos no tienen por qué
    #: repartir igual, pero sí deberían señalar al mismo entorno si la señal
    #: está en el grafo y no en la arquitectura.
    top = {}
    for modelo in ("mlp", "rf"):
        s = pd.read_csv(os.path.join(dir_modelo(modelo),
                                     "subgrafo_responsable.csv"))
        top[modelo] = s[s["rango"] == 1].set_index("id_nodo")["vecino"].to_dict()
    comunes = set(top["mlp"]) & set(top["rf"])
    iguales = sum(1 for k in comunes if top["mlp"][k] == top["rf"][k])

    for modelo in ("mlp", "rf"):
        r = json.load(open(os.path.join(dir_modelo(modelo), "resumen.json"),
                           encoding="utf-8"))
        filas.append({
            "Modelo": r["modelo"],
            "Método": "por núcleos" if modelo == "mlp" else "exacto (árboles)",
            "Valor base": f"{r['valor_base']:.4f}",
            "Error de reconstrucción": f"{r['error_maximo_reconstruccion']:.1e}",
            "Atribución al propio actor (%)": r["reparto_medio_propio"],
            "Atribución al entorno (%)": r["reparto_medio_entorno"],
            "Alertas donde manda el entorno": r["alertas_con_entorno_dominante"],
        })
    t = pd.DataFrame(filas)
    destino = os.path.join(rutas.resultados(),
                           "tabla_5_33_explicabilidad_comparada.csv")
    t.to_csv(destino, index=False)

    cotejo = {
        "correlacion_de_atribuciones": round(corr, 4),
        "alertas_en_ambos": len(comunes),
        "mismo_vecino_principal": iguales,
    }
    with open(os.path.join(SALIDA, "cotejo.json"), "w", encoding="utf-8") as fh:
        json.dump(cotejo, fh, ensure_ascii=False, indent=2)
    print("\n" + t.to_string(index=False))
    print(f"\n{json.dumps(cotejo, ensure_ascii=False, indent=2)}")
    print(f"-> {destino}")


def main():
    ap = argparse.ArgumentParser(
        description="Explicaciones de las alertas de mayor puntuación (OE-6).")
    ap.add_argument("--modelo", choices=sorted(MODELOS), default="mlp",
                    help="qué modelo se explica; por defecto el neuronal")
    ap.add_argument("--comparar", action="store_true",
                    help="explica los dos y coteja las dos explicaciones")
    ap.add_argument("--cuantas", type=int, default=100,
                    help="cuántas alertas se explican (por defecto, 100)")
    ap.add_argument("--vecinos", type=int, default=5,
                    help="cuántos vecinos se conservan por alerta")
    a = ap.parse_args()

    os.makedirs(SALIDA, exist_ok=True)
    if a.comparar:
        #: Los datos se preparan una vez y se reutilizan: construir la
        #: proyección cuesta más que explicar.
        datos = explicar("mlp", a.cuantas, a.vecinos)
        explicar("rf", a.cuantas, a.vecinos, datos)
        comparar()
    else:
        explicar(a.modelo, a.cuantas, a.vecinos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
