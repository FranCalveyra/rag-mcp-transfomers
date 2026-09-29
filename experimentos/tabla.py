"""Tabla markdown de los experimentos de la parte 1, a partir de experimentos/recuperacion/*.eval.json.

Uso: python3 experimentos/tabla.py [--orden cr|nombre]
"""
import argparse
import json
from pathlib import Path

CARPETA = Path(__file__).resolve().parent / "recuperacion"


def fmt(x):
    return "-" if x is None else f"{x:g}"


def nombre_chunking(ch):
    """Nombre de la estrategia con la terminologia de la clase (longitud fija, semantico, estructural)."""
    if ch["estrategia"] == "ventana":
        return f"longitud fija ({ch['palabras']} palabras, solap. {ch['solapamiento']})"
    return {"seccion": "estructural (sección)", "parrafo": "semántico (párrafo)",
            "oracion": "semántico (oración)"}[ch["estrategia"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orden", choices=["cr", "nombre"], default="cr")
    a = ap.parse_args()
    filas = []
    for ev in CARPETA.glob("*.jsonl.eval.json"):
        nombre = ev.name.removesuffix(".jsonl.eval.json")
        cfg = json.loads((CARPETA / f"{nombre}.config.json").read_text(encoding="utf-8"))
        r = json.loads(ev.read_text(encoding="utf-8"))["resumen"]
        filas.append((nombre, cfg["encoder"]["modelo"].split("/")[-1], nombre_chunking(cfg["chunking"]), "sí" if cfg["metadatos"] else "no",
                      cfg["top_k"], fmt(cfg["umbral"]), fmt(cfg["margen"]), "sí" if cfg.get("reranker") else "no", r))
    filas.sort(key=(lambda f: -f[-1]["context_relevance"]) if a.orden == "cr" else (lambda f: f[0]))
    print("| Encoder | Chunking | Metadatos | k | Umbral | Margen | Rerank | CR | Recall | Precision | MRR | k medio | Archivo |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nombre, enc, chunk, meta, k, u, g, rr, r in filas:
        print(f"| {enc} | {chunk} | {meta} | {k} | {u} | {g} | {rr} | **{r['context_relevance']:.3f}** | "
              f"{r['recall']:.3f} | {r['precision']:.3f} | {r['mrr']:.3f} | {r['k']:.2f} | `{nombre}` |")


if __name__ == "__main__":
    main()
