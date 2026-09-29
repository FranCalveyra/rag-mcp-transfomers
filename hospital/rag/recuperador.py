"""Recuperador vectorial: fragmenta el corpus, calcula un embedding por fragmento y devuelve los
fragmentos mas parecidos a la consulta (coseno), con top-k, umbral y margen relativo al mejor."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from hospital.rag.chunking import cargar_corpus, fragmentar
from hospital.rag.encoders import crear_encoder

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")
CACHE = RAIZ / ".cache" / "embeddings"


def seleccionar(puntuados, top_k=3, umbral=None, margen=None):
    """puntuados: [(texto, score)] ordenado de mayor a menor. Siempre devuelve al menos el mejor."""
    if not puntuados:
        return []
    mejor = puntuados[0][1]
    elegidos = [t for t, s in puntuados[:top_k]
                if (umbral is None or s >= umbral) and (margen is None or s >= mejor - margen)]
    return elegidos or [puntuados[0][0]]


class Recuperador:
    def __init__(self, docs, encoder, estrategia="seccion", palabras=60, solapamiento=15, metadatos=False,
                 top_k=3, umbral=None, margen=None, reranker=None, candidatos=20):
        self.encoder, self.metadatos = encoder, metadatos
        self.top_k, self.umbral, self.margen = top_k, umbral, margen
        self.reranker, self.candidatos = reranker, candidatos
        self.fragmentos = fragmentar(docs, estrategia, palabras=palabras, solapamiento=solapamiento)
        self.matriz = self._embeddings([f.para_embedding(metadatos) for f in self.fragmentos])

    def _embeddings(self, textos):
        nombre = getattr(self.encoder, "nombre", None)
        if nombre is None:
            return self.encoder.codificar_pasajes(textos)
        prefijo = getattr(self.encoder, "prefijo_pasaje", "")
        clave = hashlib.sha1(json.dumps([nombre, prefijo, textos]).encode()).hexdigest()[:16]
        ruta = CACHE / f"{clave}.npy"
        if ruta.exists():
            return np.load(ruta)
        m = self.encoder.codificar_pasajes(textos)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(ruta, m)
        return m

    def puntuar(self, consulta):
        """[(texto, score)] de todos los fragmentos (o de los candidatos rerankeados), de mayor a menor."""
        q = self.encoder.codificar_consultas([consulta])[0]
        scores = self.matriz @ q
        orden = np.argsort(-scores)
        if self.reranker is None:
            return [(self.fragmentos[i].texto, float(scores[i])) for i in orden]
        cand = orden[:self.candidatos]
        pares = [(consulta, self.fragmentos[i].para_embedding(self.metadatos)) for i in cand]
        re_scores = self.reranker.predict(pares, show_progress_bar=False)
        return sorted(((self.fragmentos[i].texto, float(s)) for i, s in zip(cand, re_scores)), key=lambda x: -x[1])

    def buscar(self, consulta):
        return seleccionar(self.puntuar(consulta), self.top_k, self.umbral, self.margen)

    @classmethod
    def desde_config(cls, cfg):
        """cfg: dict o ruta a un JSON como hospital/rag/config.json."""
        if not isinstance(cfg, dict):
            cfg = json.loads(Path(cfg).read_text(encoding="utf-8"))
        chunk = cfg.get("chunking", {})
        reranker = None
        if cfg.get("reranker"):
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(cfg["reranker"]["modelo"], device="cpu")
        return cls(cargar_corpus(RAIZ / cfg.get("corpus", "datos/corpus")), crear_encoder(cfg["encoder"]),
                   estrategia=chunk.get("estrategia", "seccion"), palabras=chunk.get("palabras", 60),
                   solapamiento=chunk.get("solapamiento", 15), metadatos=cfg.get("metadatos", False),
                   top_k=cfg.get("top_k", 3), umbral=cfg.get("umbral"), margen=cfg.get("margen"),
                   reranker=reranker, candidatos=(cfg.get("reranker") or {}).get("candidatos", 20))


@lru_cache(maxsize=1)
def obtener_recuperador(ruta=str(CONFIG)):
    """Instancia unica con la configuracion ganadora (la usan recuperar.py, el agente y el servidor MCP)."""
    return Recuperador.desde_config(ruta)
