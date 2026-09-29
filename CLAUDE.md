# CLAUDE.md

Misión: asistente del Hospital Arroyo Claro (RAG + agente LangChain + MCP) y ejercicios de transformers.
Consigna: `docs/mission.md`. División del trabajo: `docs/TRABAJO.md`. Contratos: `SPEC.md`.
**Plan paso a paso: `PLAN.md`** — cuando se pida "implementá el paso N", seguir esa sección.

## Comandos

```bash
source .venv/bin/activate                     # venv con Python 3.13 (torch no tiene wheel para 3.14)
pip install -r requirements.txt
python3 api/servidor.py                       # API del hospital en http://localhost:8765
pytest                                        # tests propios (tests/)
python3 recuperar.py --preguntas datos/preguntas_recuperacion_dev.jsonl --salida resultados.jsonl
python3 evaluar/evaluar.py recuperacion --preguntas datos/preguntas_recuperacion_dev.jsonl --resultados resultados.jsonl
python3 experimentos/correr_recuperacion.py   # grilla de experimentos de la parte 1
python3 experimentos/tabla.py                 # tabla markdown de experimentos
python3 atencion/test_atencion.py atencion.py
```

## Reglas

- **No modificar** `evaluar/`, `api/`, `datos/` ni `atencion/test_atencion.py`.
- La parte 5 (`a_mano/`) se resuelve a mano, sin IA: no escribir cuentas, justificaciones ni respuestas.
- Agentes con LangChain: `ChatOpenAI` apuntando a OpenRouter, tools de LangChain, sin LangGraph.
- Código compartido en el paquete `hospital/`; los scripts de la raíz son CLIs finos con los comandos del enunciado.
- Identificadores y textos en español. TDD con pytest en `tests/`. Commits convencionales, una rama por paso.
- Los fragmentos del recuperador son texto **literal** del corpus (el evaluador busca la evidencia como substring).
