"""Carga del corpus y estrategias de fragmentacion.

Todo fragmento es un substring literal del documento: el evaluador busca la evidencia como texto
dentro de los fragmentos, asi que no se reescribe, no se quitan tildes ni se agregan encabezados.
Los metadatos (titulo y seccion) viajan aparte y solo se usan, si se pide, para calcular el embedding.
"""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Documento:
    nombre: str
    texto: str

    @property
    def titulo(self):
        m = re.search(r"^# (.+)$", self.texto, re.M)
        return m.group(1).strip() if m else self.nombre


@dataclass(frozen=True)
class Fragmento:
    texto: str
    doc: str
    titulo: str
    seccion: str

    def para_embedding(self, metadatos=False):
        if not metadatos:
            return self.texto
        encabezado = f"{self.titulo} > {self.seccion}" if self.seccion else self.titulo
        return f"{encabezado}: {self.texto}"


def cargar_corpus(carpeta):
    return [Documento(p.stem, p.read_text(encoding="utf-8")) for p in sorted(Path(carpeta).glob("*.md"))]


def _secciones(doc):
    """(nombre_de_seccion, texto) por cada bloque entre encabezados; la intro va con seccion ''."""
    bloques, seccion, lineas = [], "", []
    for linea in doc.texto.splitlines():
        if linea.startswith("## "):
            bloques.append((seccion, "\n".join(lineas)))
            seccion, lineas = linea[3:].strip(), []
        elif linea.startswith("# "):
            continue
        else:
            lineas.append(linea)
    bloques.append((seccion, "\n".join(lineas)))
    return [(s, t.strip()) for s, t in bloques if t.strip()]


def _parrafos(texto):
    return [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]


def _oraciones(parrafo):
    partes = []
    for linea in parrafo.splitlines():  # los items de una lista son oraciones sueltas
        partes += re.split(r"(?<=[.!?])\s+(?=[^\sa-z])", linea.strip())
    return [p for p in partes if p]


def _ventanas(texto, palabras, solapamiento):
    spans = [m.span() for m in re.finditer(r"\S+", texto)]
    paso = palabras - solapamiento
    for i in range(0, len(spans), paso):
        grupo = spans[i:i + palabras]
        yield texto[grupo[0][0]:grupo[-1][1]]
        if i + palabras >= len(spans):
            break


def fragmentar(docs, estrategia, palabras=60, solapamiento=15):
    """estrategia: 'seccion' | 'parrafo' | 'oracion' | 'ventana' (usa palabras y solapamiento)."""
    frags = []
    for doc in docs:
        for seccion, texto in _secciones(doc):
            if estrategia == "seccion":
                piezas = [texto]
            elif estrategia == "parrafo":
                piezas = _parrafos(texto)
            elif estrategia == "oracion":
                piezas = [o for p in _parrafos(texto) for o in _oraciones(p)]
            elif estrategia == "ventana":
                piezas = list(_ventanas(texto, palabras, solapamiento))
            else:
                raise ValueError(f"estrategia desconocida: {estrategia}")
            frags += [Fragmento(p, doc.nombre, doc.titulo, seccion) for p in piezas]
    return frags
