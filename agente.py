"""Parte 2: agente con tool calling (LangChain + OpenRouter) sobre dos fuentes, los documentos del hospital
(recuperador de la parte 1) y la API del hospital (python3 api/servidor.py).

Uso: python3 agente.py --preguntas datos/preguntas_agente_dev.jsonl --salida respuestas.jsonl
Escribe respuestas.jsonl y el log de la corrida en experimentos/corridas/agente_<fecha>.md.
Necesita OPENROUTER_API_KEY (en el entorno o en .env) y la API levantada.
"""
import argparse
import asyncio
import json
from pathlib import Path

from hospital.agente.corrida import correr
from hospital.agente.modelo import crear_llm
from hospital.herramientas import HERRAMIENTAS
from hospital.rag.recuperador import obtener_recuperador


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preguntas", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    preguntas = [json.loads(l) for l in Path(a.preguntas).read_text(encoding="utf-8").splitlines() if l.strip()]
    llm = crear_llm()
    obtener_recuperador()  # carga el encoder antes de empezar, para no sumar ese tiempo a la primera pregunta
    log = asyncio.run(correr(llm, HERRAMIENTAS, preguntas, a.salida, "agente"))
    print(f"respuestas en {a.salida}, log en {log}")


if __name__ == "__main__":
    main()
