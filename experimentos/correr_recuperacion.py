"""Grilla de experimentos de la parte 1.

Por cada combinacion escribe en experimentos/recuperacion/:
  <nombre>.jsonl             resultados en el formato de recuperar.py
  <nombre>.config.json       configuracion completa (mismo esquema que hospital/rag/config.json)
  <nombre>.jsonl.eval.json   salida de evaluar/evaluar.py (sin modificar)

Uso:
  python3 experimentos/correr_recuperacion.py --encoders bert minilm --chunkings seccion parrafo --k 1 3
  python3 experimentos/correr_recuperacion.py --encoders e5-base --chunkings seccion --meta 1 --k 1 2 3 --umbral 0.8 0.82
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from hospital.rag.chunking import cargar_corpus  # noqa: E402
from hospital.rag.encoders import crear_encoder  # noqa: E402
from hospital.rag.recuperador import Recuperador, seleccionar  # noqa: E402

SALIDA = RAIZ / "experimentos" / "recuperacion"
PREGUNTAS = RAIZ / "datos" / "preguntas_recuperacion_dev.jsonl"

ENCODERS = {
    "bert": {"tipo": "bert_promedio", "modelo": "google-bert/bert-base-multilingual-cased"},
    "minilm": {"tipo": "sentence_transformer", "modelo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    "e5-small": {"tipo": "sentence_transformer", "modelo": "intfloat/multilingual-e5-small",
                 "prefijo_consulta": "query: ", "prefijo_pasaje": "passage: "},
    "e5-base": {"tipo": "sentence_transformer", "modelo": "intfloat/multilingual-e5-base",
                "prefijo_consulta": "query: ", "prefijo_pasaje": "passage: "},
    "bge-m3": {"tipo": "sentence_transformer", "modelo": "BAAI/bge-m3"},
}

CHUNKINGS = {
    "seccion": {"estrategia": "seccion"},
    "parrafo": {"estrategia": "parrafo"},
    "oracion": {"estrategia": "oracion"},
    "ventana40": {"estrategia": "ventana", "palabras": 40, "solapamiento": 10},
    "ventana80": {"estrategia": "ventana", "palabras": 80, "solapamiento": 20},
}

RERANKER = {"modelo": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "candidatos": 20}


def evaluar(resultados):
    subprocess.run([sys.executable, str(RAIZ / "evaluar" / "evaluar.py"), "recuperacion",
                    "--preguntas", str(PREGUNTAS), "--resultados", str(resultados)],
                   check=True, capture_output=True)
    return json.loads(Path(f"{resultados}.eval.json").read_text(encoding="utf-8"))["resumen"]


def fmt(x):
    return "-" if x is None else f"{x:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--chunkings", nargs="+", default=["seccion", "parrafo"], choices=list(CHUNKINGS))
    ap.add_argument("--meta", nargs="+", type=int, default=[0], choices=[0, 1])
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--umbral", nargs="+", type=float, default=[None])
    ap.add_argument("--margen", nargs="+", type=float, default=[None])
    ap.add_argument("--reranker", action="store_true")
    a = ap.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    preguntas = [json.loads(l) for l in PREGUNTAS.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = cargar_corpus(RAIZ / "datos" / "corpus")
    reranker = None
    if a.reranker:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER["modelo"], device="cpu")

    for enc_nombre in a.encoders:
        encoder = crear_encoder(ENCODERS[enc_nombre])
        for chunk_nombre, meta in itertools.product(a.chunkings, a.meta):
            chunk = CHUNKINGS[chunk_nombre]
            rec = Recuperador(docs, encoder, estrategia=chunk["estrategia"], palabras=chunk.get("palabras", 60),
                              solapamiento=chunk.get("solapamiento", 15), metadatos=bool(meta),
                              reranker=reranker, candidatos=RERANKER["candidatos"])
            puntuados = {p["id"]: rec.puntuar(p["pregunta"]) for p in preguntas}
            for k, umbral, margen in itertools.product(a.k, a.umbral, a.margen):
                nombre = f"{enc_nombre}__{chunk_nombre}__m{meta}__k{k}__u{fmt(umbral)}__g{fmt(margen)}"
                nombre += "__rr" if reranker else ""
                cfg = {"corpus": "datos/corpus", "encoder": ENCODERS[enc_nombre], "chunking": chunk,
                       "metadatos": bool(meta), "top_k": k, "umbral": umbral, "margen": margen,
                       "reranker": RERANKER if reranker else None}
                res = SALIDA / f"{nombre}.jsonl"
                res.write_text("".join(json.dumps({"id": pid, "fragmentos": seleccionar(p, k, umbral, margen)},
                                                  ensure_ascii=False) + "\n" for pid, p in puntuados.items()),
                               encoding="utf-8")
                (SALIDA / f"{nombre}.config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
                r = evaluar(res)
                print(f"{nombre:55s} CR {r['context_relevance']:.4f}  R {r['recall']:.3f}  "
                      f"P {r['precision']:.3f}  MRR {r['mrr']:.3f}  k {r['k']:.2f}", flush=True)


if __name__ == "__main__":
    main()
