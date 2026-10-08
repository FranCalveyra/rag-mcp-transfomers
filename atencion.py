"""Componentes básicos de atención implementados únicamente con NumPy."""

import numpy as np


def softmax(M):
    """Calcula softmax de cada fila, de forma numéricamente estable."""
    M = np.asarray(M)
    maximo = np.max(M, axis=-1, keepdims=True)
    exp = np.exp(M - maximo)
    return exp / np.sum(exp, axis=-1, keepdims=True)
