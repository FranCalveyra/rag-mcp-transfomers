import json
import re
import unicodedata
from pathlib import Path

import pytest

from hospital.rag.chunking import cargar_corpus, fragmentar

CORPUS = Path("datos/corpus")
PREGUNTAS = Path("datos/preguntas_recuperacion_dev.jsonl")


def norm(t):
    # misma normalizacion que evaluar/evaluar.py
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t).lower())


@pytest.fixture(scope="module")
def docs():
    return cargar_corpus(CORPUS)


def evidencias():
    return [e for l in PREGUNTAS.read_text(encoding="utf-8").splitlines() for e in json.loads(l)["evidencia"]]


def test_carga_los_20_documentos(docs):
    assert len(docs) == 20
    visitas = next(d for d in docs if d.nombre == "visitas")
    assert visitas.titulo == "Régimen de visitas"


ESTRATEGIAS = [("seccion", {}), ("parrafo", {}), ("oracion", {}), ("ventana", {"palabras": 40, "solapamiento": 10})]


@pytest.mark.parametrize("estrategia,params", ESTRATEGIAS)
def test_fragmentos_son_texto_literal_del_documento(docs, estrategia, params):
    frags = fragmentar(docs, estrategia, **params)
    fuente = {d.nombre: d.texto for d in docs}
    assert frags
    for f in frags:
        assert f.texto and f.texto in fuente[f.doc]
        assert not f.texto.lstrip().startswith("#")


@pytest.mark.parametrize("estrategia", ["seccion", "parrafo", "oracion"])
def test_ninguna_evidencia_queda_partida(docs, estrategia):
    frags = [norm(f.texto) for f in fragmentar(docs, estrategia)]
    for e in evidencias():
        assert any(norm(e) in f for f in frags), e


def test_seccion_lleva_titulo_y_nombre_de_seccion(docs):
    frags = fragmentar(docs, "seccion")
    neo = next(f for f in frags if f.doc == "visitas" and f.seccion == "Neonatología")
    assert neo.titulo == "Régimen de visitas"
    assert neo.texto.startswith("Madre y padre tienen ingreso libre")


def test_seccion_sin_subtitulos_es_el_documento_entero(docs):
    frags = [f for f in fragmentar(docs, "seccion") if f.doc == "farmacia"]
    assert len(frags) == 1 and frags[0].seccion == ""


def test_parrafo_corta_por_lineas_en_blanco(docs):
    frags = [f for f in fragmentar(docs, "parrafo") if f.doc == "farmacia"]
    assert len(frags) == 4


def test_ventana_respeta_tamano_y_solapamiento(docs):
    frags = [f for f in fragmentar(docs, "ventana", palabras=20, solapamiento=5) if f.doc == "visitas"]
    assert all(len(f.texto.split()) <= 20 for f in frags)
    pares = [(a, b) for a, b in zip(frags, frags[1:]) if a.seccion == b.seccion]  # las ventanas no cruzan secciones
    assert pares
    for a, b in pares:
        assert a.texto.split()[-5:] == b.texto.split()[:5]


def test_texto_para_embedding_con_metadatos(docs):
    f = next(f for f in fragmentar(docs, "seccion") if f.seccion == "Neonatología")
    assert f.para_embedding(metadatos=False) == f.texto
    assert f.para_embedding(metadatos=True).startswith("Régimen de visitas > Neonatología: Madre y padre")
