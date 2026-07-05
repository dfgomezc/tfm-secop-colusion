#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restaura los JSONL de origen desde los volúmenes comprimidos.

Es el reverso de `empaquetar_jsonl.py`: los JSONL crudos no forman parte de la
entrega —son datos abiertos redescargables y no hacen falta para reproducir los
resultados—, pero sí para rehacer la ingesta desde cero.

## Qué hace

Recorre los `.zip` de una carpeta, los descomprime en el destino y comprueba lo
que sale. Cada volumen guarda las rutas relativas al conjunto
—`matriculas_rues/matriculas_rues_chunk_0076.jsonl`—, de modo que descomprimir
todos en una misma carpeta reconstruye las nueve carpetas de origen tal como
estaban.

## Qué comprueba

Descomprimir puede truncar un fichero sin devolver error, y comparar tamaños no
basta. Con el manifiesto al lado se comprueba el sha256 de cada volumen antes de
abrirlo y el de cada fichero después de escribirlo. Sin manifiesto se hace lo que
se puede —el CRC que lleva el propio zip— y se avisa.

## Se puede interrumpir

Lo ya restaurado no se vuelve a escribir: un fichero que existe con el tamaño
esperado se salta. Cortar a mitad y volver a lanzar retoma donde estaba.

## Uso

    # ver qué haría, sin escribir nada
    python desempaquetar_jsonl.py --simular

    # restaurar los volúmenes que están en INPUT/, ahí mismo
    python desempaquetar_jsonl.py

    # desde otro sitio y a otro disco
    python desempaquetar_jsonl.py --zips E:/copia_jsonl --destino D:/INPUT

    # solo un conjunto
    python desempaquetar_jsonl.py --conjunto matriculas_rues

    # rehacer un fichero que se sospecha truncado
    python desempaquetar_jsonl.py --forzar --conjunto multas_sanciones_s2

    # comprobar una copia ya restaurada, sin descomprimir nada
    python desempaquetar_jsonl.py --solo-comprobar

Con `--simular` no escribe nada: se limita a informar.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tfm import rutas                                        # noqa: E402

#: Las nueve carpetas de origen. El nombre es también el prefijo del volumen
#: —`matriculas_rues_003.zip`— y la clave dentro del manifiesto.
CONJUNTOS = [
    "contratos_electronicos_s2",
    "datos_de_contacto_s2",
    "grupos_proveedores_s2",
    "matriculas_rues",
    "multas_sanciones_s2",
    "ofertas_por_proceso_s2",
    "procesos_contratacion_s2",
    "proponentes_por_proceso_s2",
    "proveedores_registrados_s2",
]

GB = 1024 ** 3

#: Donde se buscan los volúmenes por defecto: la carpeta de entrada, que es
#: donde acaban los JSONL restaurados. `rutas.entrada()` la resuelve, de modo
#: que `TFM_INPUT` sirve también aquí y los volúmenes pueden estar en otro
#: disco sin pasar `--zips`.
ZIPS_DEF = rutas.entrada()
#: El manifiesto se queda donde lo escribe `empaquetar_jsonl.py`, junto a los
#: volúmenes originales, y esa carpeta cuelga del proyecto y no del repositorio.
MANIFIESTO_DEF = rutas.zips("jsonl", "MANIFIESTO.json")


# ---------------------------------------------------------------------------
def mil(n):
    """Un entero con el punto como separador de millares."""
    return f"{n:,}".replace(",", ".")


def humano(n):
    """Un tamaño legible, con coma decimal y punto de millares."""
    for unidad in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unidad == "TB":
            return f"{n:,.1f} {unidad}".replace(",", "@").replace(".", ",").replace("@", ".")
        n /= 1024


def reloj(seg):
    """Una duración en horas, minutos y segundos, sin ceros a la izquierda."""
    seg = int(max(seg, 0))
    h, m, s = seg // 3600, (seg % 3600) // 60, seg % 60
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def conjunto_de(nombre_zip):
    """El conjunto al que pertenece un volumen, por su nombre.

    `contratos_electronicos_s2_006.zip` -> `contratos_electronicos_s2`. Se
    comprueba contra la lista en vez de cortar por el último guion bajo, que
    partiría mal cualquier nombre que no acabe en número.
    """
    base = os.path.basename(nombre_zip)
    for c in CONJUNTOS:
        if base.startswith(c + "_"):
            return c
    return None


def cargar_manifiesto(ruta):
    """El manifiesto, indexado por ruta de fichero y por nombre de volumen.

    Devuelve `(None, None)` si no está: se puede restaurar sin él, pero solo con
    el CRC del propio zip como comprobación.
    """
    if not ruta or not os.path.exists(ruta):
        return None, None
    with open(ruta, encoding="utf-8") as fh:
        m = json.load(fh)
    por_fichero = {f["ruta"]: f for v in m["volumenes"] for f in v["ficheros"]}
    por_volumen = {v["nombre"]: v for v in m["volumenes"]}
    return por_fichero, por_volumen


def seguro(rel):
    """Que una entrada del zip no se escape del destino.

    Un zip puede traer rutas absolutas o con `..`. Estos los generamos nosotros,
    pero comprobarlo cuesta tres líneas y evita que un fichero acabe fuera.
    """
    rel = rel.replace("\\", "/")
    return not (rel.startswith("/") or ".." in rel.split("/") or ":" in rel[:3])


# ---------------------------------------------------------------------------
def plan_de(zips, conjuntos, destino):
    """Los volúmenes a restaurar, en orden, con lo que pesan descomprimidos.

    Se anota aparte lo que falta por escribir —`bytes_pendientes`—, que es lo
    que de verdad tiene que caber en el disco: en una segunda pasada casi todo
    está ya y el hueco necesario es mucho menor que el total.
    """
    if not os.path.isdir(zips):
        sys.exit(f"no existe la carpeta de volúmenes: {zips}")
    fuera = []
    for nombre in sorted(os.listdir(zips)):
        if not nombre.lower().endswith(".zip"):
            continue
        c = conjunto_de(nombre)
        if c is None:
            print(f"  [!] {nombre} no corresponde a ningún conjunto conocido; "
                  f"se omite")
            continue
        if c not in conjuntos:
            continue
        ruta = os.path.join(zips, nombre)
        with zipfile.ZipFile(ruta) as zf:
            entradas = [i for i in zf.infolist() if not i.is_dir()]
            crudo = sum(i.file_size for i in entradas)
            pendiente = 0
            for i in entradas:
                p = os.path.join(destino, *i.filename.replace("\\", "/").split("/"))
                if not (os.path.exists(p) and os.path.getsize(p) == i.file_size):
                    pendiente += i.file_size
        fuera.append({"nombre": nombre, "conjunto": c, "ruta": ruta,
                      "ficheros": len(entradas), "bytes": crudo,
                      "bytes_pendientes": pendiente,
                      "bytes_zip": os.path.getsize(ruta)})
    if not fuera:
        sys.exit(f"no se encontró ningún volumen en {zips}")
    return fuera


def restaurar_volumen(vol, destino, por_fichero, forzar, estado, total_b, t0):
    """Descomprime un volumen y comprueba lo que escribe.

    Lo que ya está con el tamaño esperado no se toca, salvo `--forzar`. El
    contador de avance es global para que la estimación de tiempo restante tenga
    sentido con veintidós volúmenes por delante.
    """
    with zipfile.ZipFile(vol["ruta"]) as zf:
        entradas = [i for i in zf.infolist() if not i.is_dir()]
        for j, info in enumerate(entradas, 1):
            rel = info.filename.replace("\\", "/")
            if not seguro(rel):
                estado["rechazados"].append(rel)
                continue
            salida = os.path.join(destino, *rel.split("/"))
            esperado = por_fichero.get(rel, {}).get("bytes", info.file_size)

            if (not forzar and os.path.exists(salida)
                    and os.path.getsize(salida) == esperado):
                estado["saltados"] += 1
            else:
                os.makedirs(os.path.dirname(salida), exist_ok=True)
                #: `extract` reconstruye la ruta a partir del nombre de la
                #: entrada; se copia el flujo a mano para escribir exactamente
                #: donde toca y no depender de cómo normalice el nombre.
                with zf.open(info) as origen, open(salida, "wb") as fh:
                    shutil.copyfileobj(origen, fh, 1 << 20)
                estado["escritos"] += 1

                ficha = por_fichero.get(rel)
                if ficha:
                    if os.path.getsize(salida) != ficha["bytes"]:
                        estado["malos"].append((rel, "tamaño"))
                    elif sha256(salida) != ficha["sha256"]:
                        estado["malos"].append((rel, "sha256"))

            estado["bytes"] += info.file_size
            seg = time.time() - t0
            ritmo = estado["bytes"] / max(seg, 1e-9)
            queda = (total_b - estado["bytes"]) / max(ritmo, 1e-9)
            print(f"\r    {j}/{len(entradas)} ficheros  ·  "
                  f"{100 * estado['bytes'] / max(total_b, 1):5.1f} % del total  ·  "
                  f"{humano(ritmo)}/s  ·  faltan {reloj(queda)}    ",
                  end="", flush=True)


def desempaquetar(zips, destino, conjuntos, manifiesto, forzar, simular,
                  comprobar_zip):
    por_fichero, por_volumen = cargar_manifiesto(manifiesto)
    plan = plan_de(zips, conjuntos, destino)

    total_f = sum(v["ficheros"] for v in plan)
    total_b = sum(v["bytes"] for v in plan)
    pendiente_b = total_b if forzar else sum(v["bytes_pendientes"] for v in plan)
    print(f"\n{len(plan)} volúmenes · {mil(total_f)} ficheros · "
          f"{humano(total_b)} descomprimidos")
    for v in plan:
        print(f"  {v['nombre']:38s} {v['ficheros']:>4} ficheros  "
              f"{humano(v['bytes']):>10}")

    if por_fichero is None:
        print(f"\n  [!] sin manifiesto ({manifiesto}): se comprueba el CRC del "
              f"zip, pero no el sha256 de cada fichero")

    #: Restaurar 56 GB y quedarse sin disco a mitad es la forma más cara de
    #: descubrir que no cabía. Se mira antes, con un 2 % de margen.
    libre = shutil.disk_usage(destino if os.path.isdir(destino)
                              else os.path.dirname(os.path.abspath(destino))).free
    print(f"\ndestino  {os.path.abspath(destino)}")
    print(f"libre    {humano(libre)}   ·   hace falta {humano(pendiente_b)}"
          + (f"  (de {humano(total_b)}; el resto ya está)"
             if pendiente_b < total_b else ""))
    if libre < pendiente_b * 1.02:
        print("\n  [!] puede no caber. Usa --destino en otro disco, o restaura "
              "conjunto a conjunto con --conjunto.")
        if not simular:
            sys.exit(1)

    if simular:
        print("\n(simulación: no se ha escrito nada)")
        return 0

    os.makedirs(destino, exist_ok=True)
    estado = {"escritos": 0, "saltados": 0, "bytes": 0, "malos": [],
              "rechazados": [], "volumenes_malos": []}
    t0 = time.time()
    for k, vol in enumerate(plan, 1):
        print(f"\n[{k}/{len(plan)}] {vol['nombre']}   {vol['ficheros']} ficheros "
              f"· {humano(vol['bytes'])}", flush=True)

        ficha_vol = (por_volumen or {}).get(vol["nombre"])
        if comprobar_zip and ficha_vol:
            #: El sha256 del volumen entero: si el zip llegó mal, no tiene
            #: sentido descomprimirlo y comprobar fichero a fichero después.
            if os.path.getsize(vol["ruta"]) != ficha_vol["bytes_zip"]:
                print("    [!] el volumen no tiene el tamaño del manifiesto; se omite")
                estado["volumenes_malos"].append((vol["nombre"], "tamaño"))
                continue
            if sha256(vol["ruta"]) != ficha_vol["sha256_zip"]:
                print("    [!] el sha256 del volumen no coincide; se omite")
                estado["volumenes_malos"].append((vol["nombre"], "sha256"))
                continue

        t_vol = time.time()
        try:
            restaurar_volumen(vol, destino, por_fichero or {}, forzar, estado,
                              total_b, t0)
        except zipfile.BadZipFile as e:
            print(f"\n    [!] volumen ilegible: {e}")
            estado["volumenes_malos"].append((vol["nombre"], "ilegible"))
            continue
        print(f"\r    {vol['ficheros']} ficheros  ·  {reloj(time.time() - t_vol)}"
              f"                                          ")

    print(f"\n{'-' * 70}")
    print(f"  escritos            {mil(estado['escritos'])}")
    print(f"  ya estaban          {mil(estado['saltados'])}")
    print(f"  con problema        {len(estado['malos'])}")
    print(f"  volúmenes omitidos  {len(estado['volumenes_malos'])}")
    print(f"  tiempo              {reloj(time.time() - t0)}")
    for r, motivo in estado["malos"][:10]:
        print(f"    {motivo:8s} {r}")
    for r in estado["rechazados"][:5]:
        print(f"    ruta insegura, no se escribió: {r}")

    if estado["malos"] or estado["volumenes_malos"] or estado["rechazados"]:
        print("\nLa restauración NO está limpia. Repite con --forzar los "
              "conjuntos afectados.")
        return 1
    print("\nRestauración completa.")
    if por_fichero:
        print("Comprobado el sha256 de cada fichero escrito contra el manifiesto.")
    return 0


# ---------------------------------------------------------------------------
def comprobar(carpeta, manifiesto, conjuntos):
    """Contrasta lo que hay en el destino contra el manifiesto, sin tocar nada.

    Es la misma comprobación que `empaquetar_jsonl.py --verificar`, limitable a
    unos conjuntos: con 987 ficheros y 56 GB, recorrerlos todos para mirar uno
    cuesta más de lo que se tarda en pedirlo.
    """
    por_fichero, _ = cargar_manifiesto(manifiesto)
    if por_fichero is None:
        sys.exit(f"hace falta el manifiesto y no está en {manifiesto}")

    esperados = [f for r, f in por_fichero.items()
                 if r.split("/")[0] in conjuntos]
    print(f"{mil(len(esperados))} ficheros a comprobar en "
          f"{os.path.abspath(carpeta)}\n")

    faltan, tamano, contenido, ok = [], [], [], 0
    t0 = time.time()
    hecho_b = 0
    total_b = sum(f["bytes"] for f in esperados)
    for i, f in enumerate(esperados, 1):
        p = os.path.join(carpeta, *f["ruta"].split("/"))
        if not os.path.exists(p):
            faltan.append(f["ruta"])
        elif os.path.getsize(p) != f["bytes"]:
            tamano.append((f["ruta"], f["bytes"], os.path.getsize(p)))
        elif sha256(p) != f["sha256"]:
            contenido.append(f["ruta"])
        else:
            ok += 1
        hecho_b += f["bytes"]
        ritmo = hecho_b / max(time.time() - t0, 1e-9)
        print(f"\r  {i}/{len(esperados)}  ·  {humano(ritmo)}/s  ·  "
              f"faltan {reloj((total_b - hecho_b) / max(ritmo, 1e-9))}    ",
              end="", flush=True)

    print(f"\n\n  íntegros            {mil(ok)}")
    print(f"  ausentes            {len(faltan)}")
    print(f"  tamaño distinto     {len(tamano)}")
    print(f"  contenido distinto  {len(contenido)}")
    for r in faltan[:10]:
        print(f"    ausente  {r}")
    for r, a, b in tamano[:10]:
        print(f"    tamaño   {r}  esperado {mil(a)}, hay {mil(b)}")
    for r in contenido[:10]:
        print(f"    sha256   {r}")

    if faltan or tamano or contenido:
        print("\nLa copia NO está completa.")
        return 1
    print("\nLa copia está completa y coincide byte a byte.")
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Restaura los JSONL de origen desde los volúmenes.")
    ap.add_argument("--zips", default=ZIPS_DEF,
                    help="carpeta con los .zip (por defecto, la de entrada)")
    ap.add_argument("--destino",
                    help="dónde reconstruir las nueve carpetas "
                         "(por defecto, la misma de los .zip)")
    ap.add_argument("--conjunto", action="append", choices=CONJUNTOS,
                    help="limitar a uno o varios conjuntos; repetible")
    ap.add_argument("--manifiesto", default=MANIFIESTO_DEF,
                    help="MANIFIESTO.json, para comprobar el sha256")
    ap.add_argument("--forzar", action="store_true",
                    help="reescribir también lo que ya está restaurado")
    ap.add_argument("--sin-comprobar-zip", action="store_true",
                    help="no calcular el sha256 de cada volumen antes de abrirlo")
    ap.add_argument("--simular", action="store_true",
                    help="informar del plan sin escribir nada")
    ap.add_argument("--solo-comprobar", action="store_true",
                    help="comprobar lo ya restaurado contra el manifiesto")
    a = ap.parse_args()

    destino = a.destino or a.zips
    conjuntos = a.conjunto or CONJUNTOS

    if a.solo_comprobar:
        raise SystemExit(comprobar(destino, a.manifiesto, conjuntos))

    raise SystemExit(desempaquetar(a.zips, destino, conjuntos, a.manifiesto,
                                   a.forzar, a.simular,
                                   not a.sin_comprobar_zip))


if __name__ == "__main__":
    main()
