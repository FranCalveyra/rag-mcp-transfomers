"""Componentes básicos de atención implementados únicamente con NumPy."""

import numpy as np


def softmax(M):
    """Calcula softmax de cada fila, de forma numéricamente estable."""
    M = np.asarray(M)
    maximo = np.max(M, axis=-1, keepdims=True)
    exp = np.exp(M - maximo)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def atencion(Q, K, V, mascara=False):
    """Calcula atención de producto punto escalado.

    Devuelve tanto la salida ponderada por los valores como la matriz de
    pesos de atención.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)
    d_k = Q.shape[-1]
    puntajes = Q @ K.T / np.sqrt(d_k)

    if mascara:
        puntajes = np.where(
            np.triu(np.ones(puntajes.shape, dtype=bool), k=1),
            -np.inf,
            puntajes,
        )

    A = softmax(puntajes)
    return A @ V, A


def autoatencion(X, Wq, Wk, Wv, mascara=False):
    """Proyecta X a consultas, claves y valores y aplica atención."""
    X = np.asarray(X)
    Q = X @ np.asarray(Wq)
    K = X @ np.asarray(Wk)
    V = X @ np.asarray(Wv)
    return atencion(Q, K, V, mascara=mascara)


def multicabeza(X, cabezas, Wo, mascara=False):
    """Concatena las salidas de varias cabezas y las proyecta con ``Wo``."""
    salidas = [autoatencion(X, Wq, Wk, Wv, mascara=mascara)[0]
               for Wq, Wk, Wv in cabezas]
    concatenada = np.concatenate(salidas, axis=-1)
    return concatenada @ np.asarray(Wo)
