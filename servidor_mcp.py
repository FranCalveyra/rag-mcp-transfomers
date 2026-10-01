"""Servidor MCP stdio con las seis herramientas del hospital.

Ejecutable por MCP Inspector (python3 servidor_mcp.py) o por agente_mcp.py.
No imprimir en stdout: stdio transport reserva ese canal para el protocolo.
"""
from mcp.server.fastmcp import FastMCP

from hospital import api_cliente, documentos
from hospital.descripciones import DESCRIPCIONES

mcp = FastMCP("hospital-arroyo-claro")


@mcp.tool(description=DESCRIPCIONES["buscar_documentos"])
def buscar_documentos(consulta: str) -> str:
    return documentos.buscar_documentos(consulta)


@mcp.tool(description=DESCRIPCIONES["consultar_camas"])
def consultar_camas(sector: str) -> str:
    return api_cliente.camas(sector)


@mcp.tool(description=DESCRIPCIONES["consultar_guardia"])
def consultar_guardia(especialidad: str) -> str:
    return api_cliente.guardia(especialidad)


@mcp.tool(description=DESCRIPCIONES["consultar_turnos"])
def consultar_turnos(especialidad: str) -> str:
    return api_cliente.turnos(especialidad)


@mcp.tool(description=DESCRIPCIONES["consultar_farmacia"])
def consultar_farmacia(medicamento: str) -> str:
    return api_cliente.farmacia(medicamento)


@mcp.tool(description=DESCRIPCIONES["consultar_espera"])
def consultar_espera() -> str:
    return api_cliente.espera()


if __name__ == "__main__":
    mcp.run(transport="stdio")
