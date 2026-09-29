import asyncio
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from hospital.agente.bucle import MAX_PASOS, responder
from hospital.agente.corrida import correr, resumir


class ModeloFalso(GenericFakeChatModel):
    """Devuelve los AIMessage del guion en orden; bind_tools no cambia nada."""

    def bind_tools(self, tools, **kwargs):
        return self


def ai(contenido="", llamadas=(), entrada=10, salida=5, costo=0.001):
    return AIMessage(contenido, tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(llamadas)],
                     usage_metadata={"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida},
                     response_metadata={"token_usage": {"cost": costo}, "id": "gen-1"})


def herramienta(nombre, funcion):
    return StructuredTool.from_function(func=funcion, name=nombre, description=nombre)


CAMAS = herramienta("consultar_camas", lambda sector: json.dumps({"sector": sector, "libres": 7}))
DOCS = herramienta("buscar_documentos", lambda consulta: "[Visitas > Pediatría]\nMadre, padre o tutor pueden permanecer.")


def correr_bucle(guion, tools, pregunta="¿hay camas?"):
    return asyncio.run(responder(ModeloFalso(messages=iter(guion)), tools, pregunta))


def test_bucle_ejecuta_herramientas_y_termina_con_la_respuesta():
    msgs = correr_bucle([ai(llamadas=[("consultar_camas", {"sector": "pediatria"})]), ai("Hay 7 camas libres.")], [CAMAS])
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and '"libres": 7' in tool_msgs[0].content
    assert msgs[-1].content == "Hay 7 camas libres."


def test_herramienta_inexistente_devuelve_error_al_modelo():
    msgs = correr_bucle([ai(llamadas=[("consultar_nada", {})]), ai("No pude.")], [CAMAS])
    [tm] = [m for m in msgs if isinstance(m, ToolMessage)]
    assert "no existe" in tm.content and "consultar_camas" in tm.content


def test_error_de_la_herramienta_vuelve_como_texto():
    def falla(sector):
        raise RuntimeError("se cayó")
    msgs = correr_bucle([ai(llamadas=[("consultar_camas", {"sector": "x"})]), ai("No pude.")],
                        [herramienta("consultar_camas", falla)])
    [tm] = [m for m in msgs if isinstance(m, ToolMessage)]
    assert "se cayó" in tm.content


def test_bucle_corta_despues_de_max_pasos():
    guion = [ai(llamadas=[("consultar_camas", {"sector": "pediatria"})]) for _ in range(MAX_PASOS + 5)]
    msgs = correr_bucle(guion, [CAMAS])
    assert sum(isinstance(m, AIMessage) for m in msgs) == MAX_PASOS


def test_resumir_extrae_respuesta_contextos_herramientas_y_usage():
    msgs = correr_bucle([
        ai(llamadas=[("consultar_camas", {"sector": "pediatria"}), ("buscar_documentos", {"consulta": "acompañante"})]),
        ai(llamadas=[("consultar_camas", {"sector": "pediatria"})], costo=0.002),
        ai("Hay 7 camas y podés quedarte.", costo=0.003),
    ], [CAMAS, DOCS])
    r = resumir(msgs)
    assert r["respuesta"] == "Hay 7 camas y podés quedarte."
    assert r["herramientas"] == ["consultar_camas", "buscar_documentos"]  # sin repetir, en orden de uso
    assert len(r["contextos"]) == 3 and "Madre, padre o tutor" in r["contextos"][1]
    assert len(r["llamadas_modelo"]) == 3
    assert abs(r["costo"] - 0.006) < 1e-12 and r["tokens_entrada"] == 30


def test_resumir_normaliza_contenido_en_bloques():
    msgs = [ai(llamadas=[("x", {})]), ToolMessage(content=[{"type": "text", "text": "hola"}], tool_call_id="c0", name="x"),
            ai("fin")]
    assert resumir(msgs)["contextos"] == ["hola"]


def test_correr_escribe_jsonl_y_log(tmp_path):
    guion = [ai(llamadas=[("consultar_camas", {"sector": "pediatria"})]), ai("Hay 7 camas libres.")]
    salida = tmp_path / "respuestas.jsonl"
    log = asyncio.run(correr(ModeloFalso(messages=iter(guion)), [CAMAS],
                             [{"id": "A05", "pregunta": "¿Hay camas en pediatría?"}], salida, "prueba", carpeta_log=tmp_path))
    fila = json.loads(salida.read_text(encoding="utf-8"))
    assert fila == {"id": "A05", "respuesta": "Hay 7 camas libres.", "contextos": [fila["contextos"][0]],
                    "herramientas": ["consultar_camas"]}
    md = log.read_text(encoding="utf-8")
    for esperado in ["A05", "¿Hay camas en pediatría?", "consultar_camas", '"sector": "pediatria"', '"libres": 7',
                     "Hay 7 camas libres.", "10", "0.001"]:
        assert esperado in md, esperado


def test_las_seis_herramientas_con_los_nombres_del_evaluador():
    from hospital.descripciones import DESCRIPCIONES
    from hospital.herramientas import HERRAMIENTAS
    assert [t.name for t in HERRAMIENTAS] == ["buscar_documentos", "consultar_camas", "consultar_guardia",
                                               "consultar_turnos", "consultar_farmacia", "consultar_espera"]
    assert all(t.description == DESCRIPCIONES[t.name] for t in HERRAMIENTAS)
    assert set(HERRAMIENTAS[0].args) == {"consulta"} and HERRAMIENTAS[-1].args == {}
