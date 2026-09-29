"""Busqueda en los documentos del hospital con el recuperador de la parte 1, formateada para un LLM.

Sin LangChain: la usan la tool buscar_documentos (parte 2) y el servidor MCP (parte 3).

El recuperador es el de la parte 1 (mismo encoder, chunking e indice), pero la seleccion puede ser otra: la
clave "agente" de hospital/rag/config.json pisa top_k/umbral/margen solo para el agente. La metrica de la
parte 1 castiga cada fragmento de mas; el juez del agente, en cambio, castiga sobre todo que falte informacion
para contestar preguntas con varias partes.
"""
import json

from hospital.rag.recuperador import CONFIG, obtener_recuperador

SEPARADOR = "\n\n---\n\n"


def seleccion_agente():
    return json.loads(CONFIG.read_text(encoding="utf-8")).get("agente", {})


def buscar_documentos(consulta):
    """Fragmentos elegidos, cada uno con su fuente '[Documento > Sección]' arriba del texto literal."""
    bloques = []
    for f in obtener_recuperador().buscar_fragmentos(consulta, **seleccion_agente()):
        fuente = f"{f.titulo} > {f.seccion}" if f.seccion else f.titulo
        bloques.append(f"[{fuente}]\n{f.texto}")
    return SEPARADOR.join(bloques) or "No se encontraron documentos para esa consulta."
