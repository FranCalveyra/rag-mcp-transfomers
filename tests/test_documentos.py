from hospital import documentos
from hospital.rag.chunking import Fragmento


class RecuperadorFalso:
    def buscar_fragmentos(self, consulta, **seleccion):
        return [Fragmento("Madre y padre tienen ingreso libre.", "visitas", "Régimen de visitas", "Neonatología"),
                Fragmento("La farmacia abre a las 8.", "farmacia", "Farmacia del hospital", "")]


def test_formato_con_fuente_y_texto_literal(monkeypatch):
    monkeypatch.setattr(documentos, "obtener_recuperador", lambda: RecuperadorFalso())
    monkeypatch.setattr(documentos, "seleccion_agente", lambda: {})
    salida = documentos.buscar_documentos("visitas")
    assert "[Régimen de visitas > Neonatología]\nMadre y padre tienen ingreso libre." in salida
    assert "[Farmacia del hospital]\nLa farmacia abre a las 8." in salida
    assert salida.count("\n\n---\n\n") == 1
