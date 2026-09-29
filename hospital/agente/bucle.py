"""Bucle de tool calling con LangChain: el modelo (con las tools enlazadas con bind_tools) pide herramientas,
se ejecutan, sus resultados vuelven como ToolMessage, y se repite hasta que el modelo responde sin pedir nada.

No depende de dónde vienen las tools: sirve igual para las @tool de la parte 2 y para las que carga
langchain-mcp-adapters desde el servidor MCP en la parte 3.
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_PASOS = 6  # llamadas al modelo por pregunta

PROMPT = """Sos el asistente virtual del Hospital Provincial Arroyo Claro y respondés preguntas de pacientes y familiares.

No sabés nada de este hospital de antemano: toda la información sale de tus herramientas.
- Normas, horarios fijos, procedimientos, requisitos, preparaciones y quién puede visitar o acompañar: buscar_documentos.
- Estado de hoy (camas libres, profesionales de guardia, turnos disponibles, stock de farmacia, espera en la guardia): la herramienta de la API que corresponda.
- Si la pregunta tiene varias partes, cubrí cada una con su fuente. Por ejemplo, "¿hay lugar y me puedo quedar?" necesita las camas de hoy y la norma de acompañantes.
- Si una herramienta devuelve un error con opciones válidas, reintentá con la opción que corresponda.

Al responder:
- Usá solo datos que aparezcan en los resultados de las herramientas. No agregues horarios, nombres, cantidades ni requisitos que no estén ahí, aunque te parezcan razonables.
- Si la información no está, decilo en vez de suponer.
- Respondé en español, en pocas oraciones, contestando todo lo que se preguntó."""


async def _ejecutar(por_nombre, llamada):
    tool = por_nombre.get(llamada["name"])
    if tool is None:
        return ToolMessage(f"Error: la herramienta '{llamada['name']}' no existe. Herramientas disponibles: "
                           f"{', '.join(sorted(por_nombre))}.", tool_call_id=llamada["id"], name=llamada["name"])
    try:
        return await tool.ainvoke({**llamada, "type": "tool_call"})
    except Exception as e:  # el error vuelve al modelo como texto para que pueda corregirse
        return ToolMessage(f"Error al ejecutar {llamada['name']}: {e}", tool_call_id=llamada["id"], name=llamada["name"])


async def responder(llm, tools, pregunta):
    """Devuelve todos los mensajes de la conversación: sistema, pregunta, AIMessage y ToolMessage."""
    llm_con_tools = llm.bind_tools(tools)
    por_nombre = {t.name: t for t in tools}
    mensajes = [SystemMessage(PROMPT), HumanMessage(pregunta)]
    for _ in range(MAX_PASOS):
        ai = await llm_con_tools.ainvoke(mensajes)
        mensajes.append(ai)
        if not ai.tool_calls:
            break
        for llamada in ai.tool_calls:
            mensajes.append(await _ejecutar(por_nombre, llamada))
    return mensajes
