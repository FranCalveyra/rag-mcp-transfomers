"""Agente LangChain que descubre y usa las tools del servidor MCP por stdio."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from hospital.agente.corrida import correr
from hospital.agente.modelo import crear_llm

RAIZ = Path(__file__).resolve().parent


async def ejecutar(preguntas, salida):
    cliente = MultiServerMCPClient({
        "hospital": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(RAIZ / "servidor_mcp.py")],
            "env": {"HOSPITAL_API": os.environ.get("HOSPITAL_API", "http://localhost:8765")},
        }
    })
    async with cliente.session("hospital") as sesion:
        tools = await load_mcp_tools(sesion)  # tools/list; ainvoke realiza tools/call
        nombres = {t.name for t in tools}
        requeridas = {"buscar_documentos", "consultar_camas", "consultar_guardia", "consultar_turnos",
                      "consultar_farmacia", "consultar_espera"}
        if nombres != requeridas:
            raise RuntimeError(f"Herramientas MCP inesperadas: {sorted(nombres)}")
        return await correr(crear_llm(), tools, preguntas, salida, "agente_mcp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preguntas", required=True)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()
    preguntas = [json.loads(l) for l in Path(args.preguntas).read_text(encoding="utf-8").splitlines() if l.strip()]
    log = asyncio.run(ejecutar(preguntas, args.salida))
    print(f"respuestas en {args.salida}, log en {log}")


if __name__ == "__main__":
    main()
