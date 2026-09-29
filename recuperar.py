"""Parte 1: recuperador vectorial sobre datos/corpus/ con la configuracion fija de hospital/rag/config.json.

Uso: python3 recuperar.py --preguntas datos/preguntas_recuperacion_dev.jsonl --salida resultados.jsonl
Salida: una linea por pregunta, {"id": "R01", "fragmentos": ["texto", ...]} en orden de relevancia.
"""
import argparse
import json
from pathlib import Path

from hospital.rag.recuperador import CONFIG, obtener_recuperador


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preguntas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--config", default=str(CONFIG), help="por defecto, la configuracion ganadora")
    a = ap.parse_args()

    rec = obtener_recuperador(a.config)
    preguntas = [json.loads(l) for l in Path(a.preguntas).read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(a.salida, "w", encoding="utf-8") as f:
        for p in preguntas:
            f.write(json.dumps({"id": p["id"], "fragmentos": rec.buscar(p["pregunta"])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
