"""Corrida del benchmark: pasa cada pregunta por el agente, escribe respuestas.jsonl en el formato del evaluador
y un log .md con todo lo que hizo (llamadas a herramientas con argumentos y resultados, respuesta, y tokens y
costo de cada llamada al modelo). Lo comparten agente.py y agente_mcp.py.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from hospital.agente.bucle import responder
from hospital.agente.modelo import PRECIO_ENTRADA, PRECIO_SALIDA

RAIZ = Path(__file__).resolve().parents[2]
CORRIDAS = RAIZ / "experimentos" / "corridas"


def texto(contenido):
    """El contenido de un mensaje como texto (las tools MCP pueden devolver una lista de bloques)."""
    if isinstance(contenido, str):
        return contenido
    partes = []
    for b in contenido:
        if isinstance(b, str):
            partes.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            partes.append(b["text"])
        else:
            partes.append(json.dumps(b, ensure_ascii=False, default=str))
    return "\n".join(partes)


def _usage(ai):
    u = ai.usage_metadata or {}
    uso = (ai.response_metadata or {}).get("token_usage") or {}
    entrada, salida = u.get("input_tokens", 0), u.get("output_tokens", 0)
    costo = uso.get("cost")
    return {
        "tokens_entrada": entrada,
        "tokens_cache": (u.get("input_token_details") or {}).get("cache_read", 0),
        "tokens_salida": salida,
        "tokens_razonamiento": (u.get("output_token_details") or {}).get("reasoning", 0),
        "costo": costo if costo is not None else entrada * PRECIO_ENTRADA + salida * PRECIO_SALIDA,
        "costo_estimado": costo is None,
        "id": (ai.response_metadata or {}).get("id"),
    }


def resumir(mensajes):
    ais = [m for m in mensajes if isinstance(m, AIMessage)]
    herramientas = []
    for m in ais:
        for c in m.tool_calls:
            if c["name"] not in herramientas:
                herramientas.append(c["name"])
    llamadas = [_usage(m) for m in ais]
    respuesta = next((texto(m.content).strip() for m in reversed(ais) if texto(m.content).strip()), "")
    return {
        "respuesta": respuesta,
        "contextos": [texto(m.content) for m in mensajes if isinstance(m, ToolMessage)],
        "herramientas": herramientas,
        "llamadas_modelo": llamadas,
        "tokens_entrada": sum(l["tokens_entrada"] for l in llamadas),
        "tokens_salida": sum(l["tokens_salida"] for l in llamadas),
        "costo": sum(l["costo"] for l in llamadas),
    }


def _log_pregunta(p, mensajes, r, segundos, error):
    out = [f"## {p['id']}: {p['pregunta']}", ""]
    if error:
        out += [f"**Error:** `{error}`", ""]
    resultados = {m.tool_call_id: m for m in mensajes if isinstance(m, ToolMessage)}
    paso = 0
    for m in mensajes:
        if not isinstance(m, AIMessage):
            continue
        u = r["llamadas_modelo"][paso]
        paso += 1
        costo = f"USD {u['costo']:.6f}" + (" (estimado)" if u["costo_estimado"] else "")
        out += [f"### Llamada al modelo {paso}", "",
                f"- Tokens: entrada {u['tokens_entrada']} (cache {u['tokens_cache']}), "
                f"salida {u['tokens_salida']} (razonamiento {u['tokens_razonamiento']})",
                f"- Costo: {costo}", f"- Generación: `{u['id']}`", ""]
        if texto(m.content).strip():
            out += ["Texto del modelo:", "", "> " + texto(m.content).strip().replace("\n", "\n> "), ""]
        for c in m.tool_calls:
            res = resultados.get(c["id"])
            out += [f"**Herramienta** `{c['name']}` con argumentos `{json.dumps(c['args'], ensure_ascii=False)}`", "",
                    "```", texto(res.content) if res else "(sin resultado)", "```", ""]
    out += ["**Respuesta final:**", "", r["respuesta"] or "(vacía)", "",
            f"_Herramientas: {', '.join(r['herramientas']) or 'ninguna'} · llamadas al modelo: {len(r['llamadas_modelo'])} · "
            f"tokens {r['tokens_entrada']} / {r['tokens_salida']} · USD {r['costo']:.6f} · {segundos:.1f} s_", ""]
    return out


async def correr(llm, tools, preguntas, salida, etiqueta, carpeta_log=CORRIDAS):
    """Corre todas las preguntas y devuelve la ruta del log .md."""
    inicio = datetime.now()
    filas, secciones = [], []
    for p in preguntas:
        t0, error, mensajes = time.monotonic(), None, []
        try:
            mensajes = await responder(llm, tools, p["pregunta"])
        except Exception as e:  # una pregunta que falla no corta la corrida; queda en el log
            error = f"{type(e).__name__}: {e}"
        r = resumir(mensajes)
        seg = time.monotonic() - t0
        filas.append((p, r))
        secciones += _log_pregunta(p, mensajes, r, seg, error)
        print(f"{p['id']}  {', '.join(r['herramientas']) or '-':45s} USD {r['costo']:.5f}  {seg:4.1f}s"
              + (f"  ERROR {error}" if error else ""), flush=True)

    Path(salida).write_text("".join(
        json.dumps({"id": p["id"], "respuesta": r["respuesta"], "contextos": r["contextos"],
                    "herramientas": r["herramientas"]}, ensure_ascii=False) + "\n" for p, r in filas), encoding="utf-8")

    tot = {k: sum(r[k] for _, r in filas) for k in ["tokens_entrada", "tokens_salida", "costo"]}
    n_llamadas = sum(len(r["llamadas_modelo"]) for _, r in filas)
    cab = [f"# Corrida `{etiqueta}`: {inicio:%Y-%m-%d %H:%M:%S}", "",
           f"- Modelo: `{getattr(llm, 'model_name', type(llm).__name__)}`",
           f"- Herramientas disponibles: {', '.join(t.name for t in tools)}",
           f"- Salida: `{Path(salida).name}`",
           f"- Total: {len(filas)} preguntas, {n_llamadas} llamadas al modelo, tokens {tot['tokens_entrada']} entrada / "
           f"{tot['tokens_salida']} salida, **USD {tot['costo']:.6f}**", "",
           "| Pregunta | Herramientas | Llamadas | Tokens entrada | Tokens salida | Costo USD |",
           "|---|---|---|---|---|---|"]
    cab += [f"| {p['id']} | {', '.join(r['herramientas']) or '-'} | {len(r['llamadas_modelo'])} | {r['tokens_entrada']} | "
            f"{r['tokens_salida']} | {r['costo']:.6f} |" for p, r in filas]
    cab += [f"| **Total** | | {n_llamadas} | {tot['tokens_entrada']} | {tot['tokens_salida']} | **{tot['costo']:.6f}** |", ""]

    carpeta_log = Path(carpeta_log)
    carpeta_log.mkdir(parents=True, exist_ok=True)
    log = carpeta_log / f"{etiqueta}_{inicio:%Y%m%d-%H%M%S}.md"
    log.write_text("\n".join(cab + secciones), encoding="utf-8")
    return log
