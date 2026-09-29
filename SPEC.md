# SPEC

Contratos que fija `docs/mission.md`. La cátedra corre estos comandos tal cual.

## Modelos (OpenRouter)

| Uso | Modelo |
|---|---|
| Agentes (partes 2 y 3) | `deepseek/deepseek-v4-flash-0731` |
| Juez (`evaluar.py`) | `google/gemini-3.7-flash` |

## Parte 1: `recuperar.py`

```bash
python3 recuperar.py --preguntas <preguntas.jsonl> --salida <resultados.jsonl>
```

- Entrada: `{"id": "R01", "pregunta": "...", ...}` por línea.
- Salida: `{"id": "R01", "fragmentos": ["texto", ...]}` por línea, en orden de relevancia.
- La configuración ganadora queda fija en `hospital/rag/config.json`.
- Cada fragmento es texto literal del corpus (`datos/corpus/`).

## Parte 2: `agente.py`

```bash
python3 agente.py --preguntas <preguntas.jsonl> --salida <respuestas.jsonl>
```

- Salida: `{"id": "A01", "respuesta": "...", "contextos": ["..."], "herramientas": ["buscar_documentos", ...]}`.
- `contextos`: todo lo que devolvieron las herramientas, como texto.
- Log `.md` por corrida: pregunta, llamadas a herramientas (argumentos y resultados), respuesta, usage de cada llamada al modelo (tokens y costo).

Herramientas (nombres fijos):

| Tool | Fuente |
|---|---|
| `buscar_documentos(consulta)` | recuperador de la parte 1 |
| `consultar_camas(sector)` | `GET /camas?sector=` |
| `consultar_guardia(especialidad)` | `GET /guardia?especialidad=` |
| `consultar_turnos(especialidad)` | `GET /turnos?especialidad=` |
| `consultar_farmacia(medicamento)` | `GET /farmacia?medicamento=` |
| `consultar_espera()` | `GET /espera` |

## Parte 3: `servidor_mcp.py` y `agente_mcp.py`

```bash
python3 agente_mcp.py --preguntas <preguntas.jsonl> --salida <respuestas_mcp.jsonl>
```

- Servidor MCP stdio con el SDK `mcp`, las mismas 6 herramientas.
- Cliente LangChain con `langchain-mcp-adapters`; sin código propio de API ni recuperador.

## Parte 4: `atencion.py`

`softmax`, `atencion`, `autoatencion`, `multicabeza`, `layer_norm`, solo NumPy. Verde en `python3 atencion/test_atencion.py atencion.py`.
