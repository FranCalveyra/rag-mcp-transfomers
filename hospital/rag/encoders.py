"""Encoders transformer para la parte 1. Todos devuelven embeddings normalizados (norma L2 = 1),
asi el producto punto es el coseno.

Interfaz comun: .nombre, .codificar_consultas(textos), .codificar_pasajes(textos) -> np.ndarray
"""
import numpy as np


def _normalizar(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)


class BertPromedio:
    """Linea de base obligatoria: BERT sin ajustar para similitud, embedding = promedio de los
    vectores de la ultima capa sobre los tokens reales (se ignora el padding)."""

    def __init__(self, modelo="google-bert/bert-base-multilingual-cased", lote=16):
        from transformers import AutoModel, AutoTokenizer

        self.nombre, self.lote = modelo, lote
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModel.from_pretrained(modelo).eval()

    def _codificar(self, textos):
        import torch

        salidas = []
        with torch.no_grad():
            for i in range(0, len(textos), self.lote):
                enc = self.tokenizer(textos[i:i + self.lote], padding=True, truncation=True,
                                     max_length=512, return_tensors="pt")
                h = self.modelo(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                salidas.append(((h * m).sum(1) / m.sum(1)).numpy())
        return _normalizar(np.concatenate(salidas))

    codificar_consultas = codificar_pasajes = _codificar


class SentenceTransformerEncoder:
    """Modelos entrenados para embeddings de oraciones (MiniLM, e5, bge-m3). Los e5 piden prefijos."""

    def __init__(self, modelo, prefijo_consulta="", prefijo_pasaje="", lote=16):
        from sentence_transformers import SentenceTransformer

        self.nombre, self.lote = modelo, lote
        self.prefijo_consulta, self.prefijo_pasaje = prefijo_consulta, prefijo_pasaje
        self.modelo = SentenceTransformer(modelo, device="cpu")

    def _codificar(self, textos, prefijo):
        m = self.modelo.encode([prefijo + t for t in textos], batch_size=self.lote,
                               normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(m, dtype=np.float32)

    def codificar_consultas(self, textos):
        return self._codificar(textos, self.prefijo_consulta)

    def codificar_pasajes(self, textos):
        return self._codificar(textos, self.prefijo_pasaje)


def crear_encoder(cfg):
    """cfg: {"tipo": "bert_promedio" | "sentence_transformer", "modelo": ..., "prefijo_consulta": ..., "prefijo_pasaje": ...}"""
    if cfg["tipo"] == "bert_promedio":
        return BertPromedio(cfg["modelo"])
    if cfg["tipo"] == "sentence_transformer":
        return SentenceTransformerEncoder(cfg["modelo"], cfg.get("prefijo_consulta", ""), cfg.get("prefijo_pasaje", ""))
    raise ValueError(f"tipo de encoder desconocido: {cfg['tipo']}")
