import numpy as np

from hospital.rag.chunking import Documento
from hospital.rag.recuperador import Recuperador, seleccionar


def test_seleccionar_top_k():
    assert seleccionar([("a", 0.9), ("b", 0.8), ("c", 0.7)], top_k=2) == ["a", "b"]


def test_seleccionar_umbral_descarta_pero_deja_al_menos_uno():
    punt = [("a", 0.5), ("b", 0.4)]
    assert seleccionar(punt, top_k=3, umbral=0.45) == ["a"]
    assert seleccionar(punt, top_k=3, umbral=0.9) == ["a"]


def test_seleccionar_margen_relativo_al_mejor():
    punt = [("a", 0.9), ("b", 0.88), ("c", 0.6)]
    assert seleccionar(punt, top_k=3, margen=0.05) == ["a", "b"]


class EncoderDeJuguete:
    """Bolsa de palabras sobre un vocabulario fijo: suficiente para probar el recuperador sin modelos."""

    VOCAB = ["visita", "ayuno", "farmacia", "horas"]

    def _vec(self, textos):
        m = np.array([[t.lower().count(w) for w in self.VOCAB] for t in textos], float) + 1e-9
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    codificar_consultas = codificar_pasajes = _vec


def test_recuperador_devuelve_el_fragmento_mas_parecido():
    docs = [
        Documento("a", "# A\n\n## Visitas\n\nLa visita es a la tarde.\n\n## Ayuno\n\nEl ayuno es de 8 horas.\n"),
        Documento("b", "# B\n\nLa farmacia abre temprano.\n"),
    ]
    r = Recuperador(docs, EncoderDeJuguete(), estrategia="seccion", top_k=1)
    assert r.buscar("¿cuántas horas de ayuno?") == ["El ayuno es de 8 horas."]
    assert r.buscar("farmacia") == ["La farmacia abre temprano."]
    puntuados = r.puntuar("ayuno")
    assert [t for t, _ in puntuados][0] == "El ayuno es de 8 horas."
    assert puntuados[0][1] >= puntuados[-1][1]


def test_buscar_fragmentos_devuelve_metadatos():
    docs = [Documento("a", "# Guía A\n\n## Ayuno\n\nEl ayuno es de 8 horas.\n")]
    r = Recuperador(docs, EncoderDeJuguete(), estrategia="seccion", top_k=1)
    [f] = r.buscar_fragmentos("ayuno")
    assert (f.texto, f.titulo, f.seccion) == ("El ayuno es de 8 horas.", "Guía A", "Ayuno")


def test_buscar_fragmentos_permite_pisar_la_seleccion():
    docs = [Documento("a", "# A\n\n## Ayuno\n\nEl ayuno es de 8 horas.\n\n## Visitas\n\nLa visita dura horas.\n")]
    r = Recuperador(docs, EncoderDeJuguete(), estrategia="seccion", top_k=1)
    assert len(r.buscar_fragmentos("ayuno horas")) == 1
    assert len(r.buscar_fragmentos("ayuno horas", top_k=2, margen=None)) == 2
