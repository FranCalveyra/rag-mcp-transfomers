# División del trabajo (4 personas)

Entrega: **viernes 9 de octubre de 2026**. Consigna completa en `mission.md`.

| Persona | Bloque | Entregables | Puntos |
|---|---|---|---|
| A: _(nombre)_ | 1. RAG vectorial | `recuperar.py`, `experimentos/*.eval.json`, sección 1 del informe | 25 |
| B: _(nombre)_ | 2. Agente con tool calling (**LangChain**: `ChatOpenAI` apuntando a OpenRouter, cada herramienta es una tool de LangChain) | `agente.py`, `respuestas.jsonl` + `.eval.json`, logs `.md`, sección 2 del informe | 30 |
| C: _(nombre)_ | 3. Servidor MCP (el cliente también en **LangChain**, con `langchain-mcp-adapters`) | `servidor_mcp.py`, `agente_mcp.py`, `respuestas_mcp.jsonl` + `.eval.json`, `experimentos/inspector/`, sección 3 del informe | 15 |
| D: _(nombre)_ | 4. Atención NumPy + 5. A mano | `atencion.py`, `a_mano/` escaneado, sección 4-5 del informe | 30 |

## Dependencias

- B y C dependen de A (`buscar_documentos` llama a `recuperar.py`). Hasta que A entregue uno real, usar un stub con la misma interfaz.
- C depende de B (mismas 6 herramientas, mismo modelo, comparación de métricas y costo).
- D es independiente. La parte 4 sirve para verificar las cuentas de la parte 5.

## Reglas comunes

- No modificar `evaluar/evaluar.py`, `api/`, `datos/` ni `atencion/test_atencion.py`.
- Parte 5 a mano, sin IA.
- Commits convencionales, historia limpia, una rama por bloque.
- `INFORME.md`: una sección por dueño de bloque, más el costo total en OpenRouter (lo consolida A o quien se ofrezca).
