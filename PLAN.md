# Plan: Hospital Arroyo Claro — RAG, LangChain agent, MCP, Transformers

## Context

`docs/mission.md` asks for 5 exercises due **2026-10-09**. The repo holds only what the course gave us:
`datos/` (20-doc corpus + dev questions), `api/servidor.py` (hospital API on :8765, stdlib only), `evaluar/evaluar.py`
(official evaluator), `atencion/test_atencion.py` (14 tests), `a_mano/ejercicio.md`, `docs/TRABAJO.md` (4-person split).
**None of the deliverables exist yet**: `recuperar.py`, `agente.py`, `servidor_mcp.py`, `agente_mcp.py`, `atencion.py`,
`experimentos/`, `INFORME.md`, `CLAUDE.md`, `SPEC.md`.

This plan is written so a session can be told **"implement step N"** and execute that step alone. It lives at `PLAN.md` in the repo root
(referenced from `CLAUDE.md` once Step 0 creates it), so future sessions find it.

The agent (parts 2 and 3) is built with **LangChain**, as the mission requires: the model is `langchain_openai.ChatOpenAI`
pointed at OpenRouter, every tool is a LangChain tool (`@tool` in part 2, loaded with `langchain-mcp-adapters` in part 3),
and the tool-calling loop is a small explicit loop over `llm.bind_tools(tools)` (no LangGraph).

### Facts discovered in the repo that shape the design
- **Never modify** `evaluar/`, `api/`, `datos/`, `atencion/test_atencion.py`.
- Retrieval metric: each dev question has **one short verbatim evidence phrase** (e.g. `"de 12:00 a 12:30"`). The
  evaluator checks `evidence in fragment` after only `NFKC + lower + collapse whitespace`. ⇒ fragments must be **verbatim
  corpus text** (no accent stripping or rewriting) and precision = share of returned fragments containing the phrase, so
  the ideal is **few, small fragments** (k≈1–2) that never split a phrase.
- Corpus: 20 short Markdown docs (~2.5k words total), `# Title` + `## Section` + one paragraph per section ⇒ section-
  or paragraph-level chunking is the natural candidate.
- Evaluator writes `<results file>.eval.json` next to the results file.
- API is a fixed snapshot (fecha 2026-10-05). Valid names: camas {clinica_medica, cirugia_general, terapia_intensiva,
  pediatria, neonatologia, maternidad}; guardia {clinica_medica, cardiologia, pediatria, traumatologia, obstetricia,
  salud_mental}; turnos {cardiologia, dermatologia, traumatologia, psicologia, gastroenterologia, obstetricia,
  kinesiologia}; farmacia {amoxicilina 500 mg, enalapril 10 mg, metformina 850 mg, salbutamol aerosol, insulina NPH,
  paracetamol 500 mg, levotiroxina 50 mcg}; espera levels {rojo, naranja, amarillo, verde, azul}. Unknown names return
  an error **with the valid options** (lets the agent self-correct). Names are accent/case-insensitive, space = `_`.
- Agent dev questions: A01–A04 docs only, A05–A09 one API tool each, A10–A12 API + docs.
- Local Python is **3.14**; if `torch` has no wheel, create the venv with 3.12 (`uv venv -p 3.12` or `python3.12 -m venv`).

## Target layout (built up across steps)

```
CLAUDE.md  SPEC.md  INFORME.md  PLAN.md
requirements.txt                 # + pytest
hospital/                        # shared code (package)
  rag/chunking.py  rag/encoders.py  rag/recuperador.py  rag/config.json
  api_cliente.py                 # 6 plain functions → JSON text (urllib, no LangChain)
  descripciones.py               # tool names + descriptions, shared by agente.py and servidor_mcp.py
  agente/bucle.py                # LangChain tool-calling loop (tool-source agnostic)
  agente/corrida.py              # run questions → respuestas.jsonl + .md log (shared by both agents)
recuperar.py  agente.py  servidor_mcp.py  agente_mcp.py  atencion.py
experimentos/  correr_recuperacion.py  tabla.py  comparar_agentes.py  *.jsonl(.eval.json)  corridas/*.md  inspector/*.png
tests/  test_chunking.py  test_recuperador.py  test_api_cliente.py  test_corrida.py
a_mano/  verificar.py  (+ scanned sheets)
```

## Conventions for every step
- TDD where it makes sense (pytest in `tests/`), conventional commits, one branch per step (`parte-N-<slug>`), per `docs/TRABAJO.md`.
- Code identifiers and user-facing text in Spanish (matches the course); keep scripts runnable with the exact commands of the mission.
- Each step ends by filling its section of `INFORME.md` and passing its **Done when** checks.
- Don't tune to the dev set by per-question hacks: the course grades on a hidden test set of the same shape.

---

## Step 0 — Project setup (prerequisite for all)
1. Create venv, `pip install -r requirements.txt`; add `pytest` to `requirements.txt`.
2. `.gitignore`: `.venv/`, `.cache/`, `.env`, `__pycache__/`.
3. `CLAUDE.md`: commands (API, evaluator, tests), “do not modify” list, pointer to `PLAN.md`, conventions above.
4. `SPEC.md`: the 5 CLI contracts and JSONL formats (copied from the mission), tool names/signatures, model ids.
5. `INFORME.md` skeleton: sections 1–5 + “Costo total”.
6. Reference `PLAN.md` from `CLAUDE.md`. Create empty `hospital/`, `tests/`, `experimentos/` packages/dirs.

**Done when:** `python3 api/servidor.py` answers `curl "localhost:8765/camas?sector=pediatria"`; `pytest` runs (0 tests OK).

---

## Step 1 — Part 1: vector RAG (`recuperar.py`, 25 pts)

**1.1 Chunking** — `hospital/rag/chunking.py`
- `Fragmento(texto, doc, titulo, seccion)` dataclass. Strategies: `por_seccion` (split on `##`, text = section body),
  `por_parrafo`, `por_oracion`, `ventana(n_palabras, solapamiento)`.
- `texto` must be verbatim corpus text (substring-preserving). Optional `con_metadatos`: embed `"{titulo} > {seccion}: {texto}"` but **return** only `texto`.
- Tests: every dev evidence phrase lands wholly inside ≥1 fragment for `por_seccion`/`por_parrafo`; fragments are substrings of the source doc.

**1.2 Encoders** — `hospital/rag/encoders.py`, common interface `codificar_consultas(list[str])` / `codificar_pasajes(list[str])` → L2-normalized `np.ndarray`.
- `BertPromedio(modelo)`: `transformers.AutoModel`, **mean of last-layer token vectors weighted by attention mask** (mandatory baseline, `google-bert/bert-base-multilingual-cased`).
- `SentenceTransformerEncoder(modelo, prefijo_consulta="", prefijo_pasaje="")`: MiniLM-L12-v2, e5-small, e5-base (`query: ` / `passage: `), optionally `BAAI/bge-m3` (heavy).
- Cache passage embeddings in `.cache/` keyed by (model, chunking, metadata) hash.

**1.3 Retriever** — `hospital/rag/recuperador.py`
- `Recuperador(config)` → `.buscar(consulta) -> list[str]`: cosine top-k, drop below `umbral`, optional relative cutoff (`score >= mejor - margen`), always return ≥1.
- `Recuperador.desde_config("hospital/rag/config.json")` loads the frozen winner; lazily-built singleton so the agent/MCP server loads the model once.

**1.4 Experiments** — `experimentos/correr_recuperacion.py`
- Grid defined in code: encoders × chunkings × metadata on/off × k ∈ {1,2,3} × umbral. For each config `nombre`: write
  `experimentos/<nombre>.jsonl` + `<nombre>.config.json`, run `evaluar/evaluar.py recuperacion` via subprocess (→ `<nombre>.jsonl.eval.json`).
- Order: baseline BERT first; then find best chunking with one good encoder; then compare ≥3 encoders on that chunking; then tune k/umbral; optional cross-encoder rerank (e.g. `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) last.
- `experimentos/tabla.py`: builds the markdown table (encoder, chunking, metadatos, k, umbral, CR, recall, precision, MRR) from all `.eval.json`.

**1.5 Freeze + CLI**
- Write winner to `hospital/rag/config.json`. `recuperar.py --preguntas --salida` uses only that config.
- INFORME §1: table, chosen config, why the encoder won (numbers: CR/recall/precision vs BERT baseline, e.g. anisotropy of raw BERT means).

**Done when:** `python3 recuperar.py --preguntas datos/preguntas_recuperacion_dev.jsonl --salida resultados.jsonl && python3 evaluar/evaluar.py recuperacion --preguntas datos/preguntas_recuperacion_dev.jsonl --resultados resultados.jsonl` clearly beats the BERT baseline and the naive 0.35; every table row has its `.eval.json`.

---

## Step 2 — Part 2: LangChain agent with two sources (`agente.py`, 30 pts)
Depends on Step 1 (until then, `buscar_documentos` may use a stub with the same interface).

**2.1 API client** — `hospital/api_cliente.py`: `camas(sector)`, `guardia(especialidad)`, `turnos(especialidad)`,
`farmacia(medicamento)`, `espera()` → JSON **string** (including error bodies with `opciones`, so the model can retry);
base URL from env `HOSPITAL_API` default `http://localhost:8765`. Test against the running API.

**2.2 Descriptions** — `hospital/descripciones.py`: one dict `{nombre: descripcion}` for the 6 tools. Each says what the
source contains, when to use it (static norms/procedures → `buscar_documentos`; today’s live state → API), the valid
parameter values, and that mixed questions need both. Shared by Step 3 so the comparison is fair.

**2.3 LangChain tools** — in `agente.py` (or `hospital/herramientas_lc.py`): 6 `@tool`s with the exact names
`buscar_documentos(consulta)`, `consultar_camas(sector)`, `consultar_guardia(especialidad)`, `consultar_turnos(especialidad)`,
`consultar_farmacia(medicamento)`, `consultar_espera()`, `description=DESCRIPCIONES[...]`. `buscar_documentos` returns
the fragments of `Recuperador.desde_config()` joined with `\n\n---\n\n`.

**2.4 Tool-calling loop** — `hospital/agente/bucle.py`, `async responder(llm, tools, pregunta) -> list[BaseMessage]`:
```python
llm_t = llm.bind_tools(tools)
por_nombre = {t.name: t for t in tools}
mensajes = [SystemMessage(PROMPT), HumanMessage(pregunta)]
for _ in range(MAX_PASOS):                      # ≈6 model calls
    ai = await llm_t.ainvoke(mensajes); mensajes.append(ai)
    if not ai.tool_calls:
        break
    for call in ai.tool_calls:
        mensajes.append(await por_nombre[call["name"]].ainvoke(call))   # returns a ToolMessage
return mensajes
```
Unknown tool names or tool exceptions become a `ToolMessage` with the error text, so the model can correct itself.
Async everywhere (MCP tools in Step 3 are async-only). System prompt (Spanish): answer only from tool results, never
invent data, call every source the question needs (docs + API for mixed ones), retry with a valid option on error,
short complete answer. `MAX_PASOS≈6`.
LLM: `ChatOpenAI(model="deepseek/deepseek-v4-flash-0731", base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, temperature=0, extra_body={"usage": {"include": True}})`.

**2.5 Run + log** — `hospital/agente/corrida.py`, `async correr(llm, tools, preguntas, salida, etiqueta)`:
- per question `responder(...)`, then from the messages: `herramientas` = ordered unique names of `AIMessage.tool_calls`;
  `contextos` = text of every `ToolMessage` (normalize MCP content blocks to text); `respuesta` = last AIMessage content.
- Writes `respuestas.jsonl` and `experimentos/corridas/<etiqueta>_<timestamp>.md`: per question → question, each model
  call (input/output tokens from `usage_metadata`, cost from `response_metadata["token_usage"]["cost"]`, fallback
  computed with 0.04/0.64 USD per M), each tool call with args and full result, final answer; run totals at the end.
- Tests with a fake chat model that emits tool calls (verifies extraction of herramientas/contextos and log format, no network).

**2.6 Evaluate + iterate** — run the benchmark, then `evaluar.py agente` (→ `respuestas.jsonl.eval.json`). Iterate on
descriptions/prompt (not per-question hacks) until ruteo≈1 and CR/F/AR > 4. Keep every run’s log + eval.
INFORME §2: metrics, cost, analysis of failed questions citing the logs.

**Done when:** `python3 agente.py --preguntas datos/preguntas_agente_dev.jsonl --salida respuestas.jsonl` produces the jsonl + md log; evaluator shows ruteo ≈ 1 and three metrics > 4.

---

## Step 3 — Part 3: MCP server + LangChain MCP client (15 pts)
Depends on Step 2.

**3.1 `servidor_mcp.py`** — `mcp.server.fastmcp.FastMCP("hospital")`, 6 `@mcp.tool(name=..., description=DESCRIPCIONES[...])`
wrapping `hospital.api_cliente` and `Recuperador`; `mcp.run()` (stdio). Nothing on stdout except the protocol (logs → stderr);
warm up the retriever at startup so the first call doesn’t time out.

**3.2 `agente_mcp.py`** — **no API/retriever code of its own**: `MultiServerMCPClient({"hospital": {"command": sys.executable, "args": ["servidor_mcp.py"], "transport": "stdio"}})`,
`tools = await client.get_tools()` (tools/list; log the discovered names), then reuse `responder` + `correr(..., etiqueta="mcp")`
with the same LLM and prompt.

**3.3 Inspector** (manual, user) — `npx @modelcontextprotocol/inspector python3 servidor_mcp.py`, call all 6 tools, save screenshots to `experimentos/inspector/`.

**3.4 Compare** — run benchmark → `respuestas_mcp.jsonl` + log + eval. `experimentos/comparar_agentes.py` prints a table
(ruteo, CR, F, AR, agent cost from log totals, judge cost) for Part 2 vs 3. INFORME §3: table + explanation of any difference from the logs (e.g. tool-result formatting, nondeterminism).

**Done when:** `python3 agente_mcp.py --preguntas datos/preguntas_agente_dev.jsonl --salida respuestas_mcp.jsonl` works with the API running, `grep -E "urllib|Recuperador|localhost" agente_mcp.py` is empty, 6 inspector screenshots exist.

---

## Step 4 — Part 4: attention in NumPy (`atencion.py`, 15 pts)
Independent. TDD against the given tests (red → green per function):
- `softmax(M)`: subtract row max, exp, normalize over last axis.
- `atencion(Q, K, V, mascara=False)`: `S = Q Kᵀ / √d_k`; if mask, `np.triu(ones, 1)` → `-inf`; `A = softmax(S)`; return `(A V, A)`.
- `autoatencion(X, Wq, Wk, Wv, mascara)`: project then `atencion`.
- `multicabeza(X, cabezas, Wo, mascara)`: concat head outputs on axis 1, `@ Wo`.
- `layer_norm(x, eps=1e-5)`: `(x - mean) / sqrt(var + eps)` per row, no γ/β.
NumPy only. INFORME §4 short.

**Done when:** `python3 atencion/test_atencion.py atencion.py` → 14 tests OK (test file untouched).

---

## Step 5 — Part 5: transformer block by hand (15 pts) — support only
The calculations, the per-operation justifications and the answers to the 5 questions **must be done by hand, without AI**
(mission rule). The session only builds support:
- `a_mano/verificar.py`: uses `atencion.py` and the data of `a_mano/ejercicio.md` to print, rounded to 3 decimals, each
  intermediate value (X, Q/K/V, S, S/√d, masked S, A, AV·Wo, Z, LN(Z), FFN, H, LN(H), logits, softmax) for both
  sentences, with and without mask — to be run **after** the hand calculation to check it.
- `a_mano/README.md`: checklist of sheets to scan (naming `hoja_01.jpg`…), Parts A/B and 5 questions covered.
- INFORME §5: pointer to the sheets (content written by the team).

**Done when:** `python3 a_mano/verificar.py` runs; sheets scanned into `a_mano/` by the team.

---

## Step 6 — Report consolidation & delivery
- INFORME “Costo total”: sum agent costs from all `experimentos/corridas/*.md` + `costo_juez_usd` from all agent `.eval.json`; contrast with a screenshot/figure of the OpenRouter activity dashboard; note any model-id substitution.
- Delivery checklist against mission “La entrega”: the 5 scripts run with the exact commands; `experimentos/` evals; `respuestas*.jsonl` + `.eval.json` + logs; inspector screenshots; untouched test file green; `a_mano/` scans; `git status` clean on protected dirs (`git diff main --stat -- evaluar api datos atencion` empty).

## Verification summary
| Step | Command |
|---|---|
| 1 | `python3 recuperar.py ... && python3 evaluar/evaluar.py recuperacion ...` |
| 2 | API running + `OPENROUTER_API_KEY`; `python3 agente.py ...`; `python3 evaluar/evaluar.py agente --respuestas respuestas.jsonl` |
| 3 | `python3 agente_mcp.py ...`; evaluator on `respuestas_mcp.jsonl`; MCP Inspector |
| 4 | `python3 atencion/test_atencion.py atencion.py` |
| all | `pytest` |
