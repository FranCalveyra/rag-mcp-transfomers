"""Las seis herramientas de la parte 2 como tools de LangChain. Los nombres son los que exige el evaluador y
las descripciones salen de hospital/descripciones.py (las mismas que usa el servidor MCP)."""
from langchain_core.tools import tool

from hospital import api_cliente, documentos
from hospital.descripciones import DESCRIPCIONES


@tool("buscar_documentos", description=DESCRIPCIONES["buscar_documentos"])
def buscar_documentos(consulta: str) -> str:
    return documentos.buscar_documentos(consulta)


@tool("consultar_camas", description=DESCRIPCIONES["consultar_camas"])
def consultar_camas(sector: str) -> str:
    return api_cliente.camas(sector)


@tool("consultar_guardia", description=DESCRIPCIONES["consultar_guardia"])
def consultar_guardia(especialidad: str) -> str:
    return api_cliente.guardia(especialidad)


@tool("consultar_turnos", description=DESCRIPCIONES["consultar_turnos"])
def consultar_turnos(especialidad: str) -> str:
    return api_cliente.turnos(especialidad)


@tool("consultar_farmacia", description=DESCRIPCIONES["consultar_farmacia"])
def consultar_farmacia(medicamento: str) -> str:
    return api_cliente.farmacia(medicamento)


@tool("consultar_espera", description=DESCRIPCIONES["consultar_espera"])
def consultar_espera() -> str:
    return api_cliente.espera()


HERRAMIENTAS = [buscar_documentos, consultar_camas, consultar_guardia, consultar_turnos, consultar_farmacia,
                consultar_espera]
