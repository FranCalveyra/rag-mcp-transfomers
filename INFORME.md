# Informe: RAG, MCP y Transformers en el Hospital Arroyo Claro

## 1. RAG vectorial

### Configuración entregada

`hospital/rag/config.json`, que usa `recuperar.py`:

| Encoder | Chunking | Metadatos | Selección | CR dev | Recall | Precision | MRR |
|---|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-base` (`query: ` / `passage: `) | por sección `##` | título > sección antepuestos al embedding | top-2 con margen 0,005 respecto del mejor | **0,967** | 1,000 | 0,950 | 1,000 |

Línea de base obligatoria (BERT multilingüe sin ajustar, promedio de la última capa): **0,350** en su mejor
configuración, igual al recuperador léxico ingenuo del enunciado. La configuración entregada casi triplica ese valor.

### Cómo se eligió

1. **Encoders × chunking × metadatos × k ∈ {1, 3}** para los cuatro encoders (80 corridas). Cada pregunta dev tiene una
   sola frase de evidencia, así que devolver un fragmento extra baja la precisión a la mitad: con k = 3 ningún encoder
   pasa de 0,515. Por eso el ajuste fino se hizo con k chico.
2. **Metadatos:** anteponer "título del documento > sección" al texto del embedding (el fragmento devuelto sigue siendo
   literal) mejora a los cuatro encoders con chunking por sección y k = 1: BERT 0,25 → 0,35, MiniLM 0,70 → 0,90,
   e5-small 0,80 → 0,95, e5-base 0,85 → 1,00. La única combinación donde empeora es BERT con ventanas de 40 palabras
   (0,30 → 0,20). Muchas secciones no nombran su tema ("La unidad tiene dos franjas de visita…"): el tema está en el encabezado.
3. **Chunking:** sección y ventana de 80 palabras empatan arriba. Oración es la peor para todos: corta el contexto que
   el encoder necesita para reconocer el tema. Se eligió sección porque respeta la estructura del documento y da
   fragmentos completos (útil para el agente de la parte 2, p. ej. la preparación de la colonoscopía entera).
4. **Selección (top-k, umbral, margen):** con e5-base + sección + metadatos, k = 1 da 1,000 en dev. Pero en dos preguntas
   el primer y el segundo fragmento están casi empatados (R01: diferencia de 0,001 en el coseno; R15: 0,003). En el
   conjunto de test un empate así es casi una moneda al aire: con k = 1 un error vale 0, y devolver los dos vale 0,667.
   Por eso se entrega **top-2 con margen 0,005**: el segundo fragmento se agrega solo si está a menos de 0,005 del
   primero (k medio 1,10). Cuesta 0,033 en dev y protege los casos dudosos. Un umbral absoluto de 0,88 saca más en dev
   (0,983), pero no cubre esos casos: agrega el segundo fragmento cuando los dos puntajes son altos, no cuando están
   empatados, y en R01 (0,862) y R15 (0,816) el mejor ni siquiera llega a 0,88. Como los cosenos de e5 viven en una franja
   estrecha (el mejor va de 0,82 a 0,91 según la pregunta), un umbral absoluto depende demasiado del conjunto dev; el
   margen es relativo a cada pregunta.

### Por qué ganó e5-base

| Encoder | Mejor CR | CR medio con k = 1 (10 configs) | Coseno medio entre fragmentos | z del fragmento correcto |
|---|---|---|---|---|
| BERT multilingüe (promedio) | 0,350 | 0,27 | 0,786 | 1,56 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0,950 | 0,76 | 0,365 | 3,07 |
| multilingual-e5-small | 0,950 | 0,80 | 0,842 | 3,94 |
| **multilingual-e5-base** | **1,000** | **0,85** | 0,830 | 3,94 |

*z del fragmento correcto*: cuántos desvíos estándar está el coseno del fragmento con la evidencia por encima del
promedio de todos los fragmentos para esa pregunta (secciones con metadatos, promedio sobre las 20 preguntas).

- BERT sin ajustar no fue entrenado para que el coseno mida similitud de significado: el promedio de sus tokens
  refleja sobre todo la forma superficial del texto, y el fragmento correcto apenas se separa del resto (z = 1,56).
- Que los vectores estén "amontonados" no es el problema en sí: e5 tiene cosenos medios tan altos como BERT (0,83),
  pero fue entrenado con pares consulta–pasaje (por eso los prefijos `query:` / `passage:`), así que ordena bien dentro
  de esa franja: el fragmento correcto queda a 3,94 desvíos. MiniLM está entrenado para paráfrasis entre oraciones, no
  para preguntas contra pasajes, y separa menos (3,07).
- e5-base y e5-small separan igual en promedio, pero e5-base acierta el primer lugar en más configuraciones (CR medio
  0,85 contra 0,80) y es el único que llega a 1,00 con k = 1. Corre en CPU en pocos segundos para las 56 secciones.

### Tabla de experimentos

Una fila por configuración (98). Cada una tiene en `experimentos/recuperacion/` su `<archivo>.jsonl`,
`<archivo>.config.json` y `<archivo>.jsonl.eval.json`. Se regenera con `python3 experimentos/tabla.py`, y se corre con
`python3 experimentos/correr_recuperacion.py` (ver `--help`). Ordenada por CR.

| Encoder | Chunking | Metadatos | k | Umbral | Margen | Rerank | CR | Recall | Precision | MRR | k medio | Archivo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-base | ventana 80/20 | sí | 1 | - | - | no | **1.000** | 1.000 | 1.000 | 1.000 | 1.00 | `e5-base__ventana80__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 1 | - | - | no | **1.000** | 1.000 | 1.000 | 1.000 | 1.00 | `e5-base__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 3 | 0.88 | - | no | **0.983** | 1.000 | 0.975 | 1.000 | 1.05 | `e5-base__seccion__m1__k3__u0.88__g-` |
| multilingual-e5-base | seccion | sí | 2 | 0.88 | - | no | **0.983** | 1.000 | 0.975 | 1.000 | 1.05 | `e5-base__seccion__m1__k2__u0.88__g-` |
| multilingual-e5-base | seccion | sí | 3 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__seccion__m1__k3__u-__g0.005` |
| multilingual-e5-base | ventana 80/20 | sí | 2 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__ventana80__m1__k2__u-__g0.005` |
| multilingual-e5-base | seccion | sí | 2 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__seccion__m1__k2__u-__g0.005` |
| multilingual-e5-base | ventana 80/20 | sí | 3 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__ventana80__m1__k3__u-__g0.005` |
| multilingual-e5-base | ventana 80/20 | sí | 2 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__ventana80__m1__k2__u-__g0.01` |
| multilingual-e5-base | seccion | sí | 3 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__seccion__m1__k3__u-__g0.01` |
| multilingual-e5-small | seccion | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-small__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | parrafo | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-base__parrafo__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 2 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__seccion__m1__k2__u-__g0.01` |
| multilingual-e5-base | ventana 80/20 | sí | 3 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__ventana80__m1__k3__u-__g0.01` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 80/20 | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `minilm__ventana80__m1__k1__u-__g-` |
| multilingual-e5-small | ventana 80/20 | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-small__ventana80__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | parrafo | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `minilm__parrafo__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 2 | 0.85 | - | no | **0.900** | 1.000 | 0.850 | 1.000 | 1.30 | `e5-base__seccion__m1__k2__u0.85__g-` |
| multilingual-e5-small | parrafo | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `e5-small__parrafo__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | seccion | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `minilm__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 2 | - | 0.02 | no | **0.900** | 1.000 | 0.850 | 1.000 | 1.30 | `e5-base__seccion__m1__k2__u-__g0.02` |
| multilingual-e5-base | ventana 80/20 | sí | 2 | - | 0.02 | no | **0.883** | 1.000 | 0.825 | 1.000 | 1.35 | `e5-base__ventana80__m1__k2__u-__g0.02` |
| multilingual-e5-base | seccion | sí | 3 | 0.85 | - | no | **0.867** | 1.000 | 0.817 | 1.000 | 1.50 | `e5-base__seccion__m1__k3__u0.85__g-` |
| multilingual-e5-base | seccion | sí | 3 | - | 0.02 | no | **0.867** | 1.000 | 0.817 | 1.000 | 1.50 | `e5-base__seccion__m1__k3__u-__g0.02` |
| multilingual-e5-base | ventana 40/10 | sí | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__ventana40__m1__k1__u-__g-` |
| multilingual-e5-base | seccion | no | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__seccion__m0__k1__u-__g-` |
| multilingual-e5-base | ventana 80/20 | no | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__ventana80__m0__k1__u-__g-` |
| multilingual-e5-base | ventana 80/20 | sí | 3 | - | 0.02 | no | **0.850** | 1.000 | 0.792 | 1.000 | 1.55 | `e5-base__ventana80__m1__k3__u-__g0.02` |
| multilingual-e5-base | parrafo | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | parrafo | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | seccion | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__seccion__m0__k1__u-__g-` |
| multilingual-e5-base | ventana 40/10 | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__ventana40__m0__k1__u-__g-` |
| multilingual-e5-base | oracion | sí | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__oracion__m1__k1__u-__g-` |
| multilingual-e5-small | ventana 80/20 | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__ventana80__m0__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 2 | 0.82 | - | no | **0.750** | 1.000 | 0.625 | 1.000 | 1.75 | `e5-base__seccion__m1__k2__u0.82__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | parrafo | no | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | ventana 40/10 | no | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__ventana40__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 40/10 | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__ventana40__m1__k1__u-__g-` |
| multilingual-e5-small | oracion | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__oracion__m1__k1__u-__g-` |
| multilingual-e5-small | ventana 40/10 | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__ventana40__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | oracion | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__oracion__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | seccion | no | 1 | - | - | no | **0.700** | 0.700 | 0.700 | 0.700 | 1.00 | `minilm__seccion__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 80/20 | no | 1 | - | - | no | **0.700** | 0.700 | 0.700 | 0.700 | 1.00 | `minilm__ventana80__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 40/10 | no | 1 | - | - | no | **0.650** | 0.650 | 0.650 | 0.650 | 1.00 | `minilm__ventana40__m0__k1__u-__g-` |
| multilingual-e5-base | seccion | sí | 3 | 0.82 | - | no | **0.633** | 1.000 | 0.508 | 1.000 | 2.45 | `e5-base__seccion__m1__k3__u0.82__g-` |
| multilingual-e5-base | oracion | no | 1 | - | - | no | **0.600** | 0.600 | 0.600 | 0.600 | 1.00 | `e5-base__oracion__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | oracion | no | 1 | - | - | no | **0.550** | 0.550 | 0.550 | 0.550 | 1.00 | `minilm__oracion__m0__k1__u-__g-` |
| multilingual-e5-small | oracion | no | 1 | - | - | no | **0.550** | 0.550 | 0.550 | 0.550 | 1.00 | `e5-small__oracion__m0__k1__u-__g-` |
| multilingual-e5-small | ventana 80/20 | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.967 | 3.00 | `e5-small__ventana80__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 80/20 | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.975 | 3.00 | `minilm__ventana80__m1__k3__u-__g-` |
| multilingual-e5-base | ventana 80/20 | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 1.000 | 3.00 | `e5-base__ventana80__m1__k3__u-__g-` |
| multilingual-e5-base | ventana 40/10 | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.925 | 3.00 | `e5-base__ventana40__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 40/10 | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.875 | 3.00 | `minilm__ventana40__m1__k3__u-__g-` |
| multilingual-e5-base | parrafo | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.967 | 3.00 | `e5-base__parrafo__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | seccion | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.950 | 3.00 | `minilm__seccion__m1__k3__u-__g-` |
| multilingual-e5-small | parrafo | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.942 | 3.00 | `e5-small__parrafo__m1__k3__u-__g-` |
| multilingual-e5-base | seccion | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 1.000 | 3.00 | `e5-base__seccion__m1__k3__u-__g-` |
| multilingual-e5-small | seccion | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.967 | 3.00 | `e5-small__seccion__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | parrafo | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.950 | 3.00 | `minilm__parrafo__m1__k3__u-__g-` |
| multilingual-e5-small | ventana 40/10 | sí | 3 | - | - | no | **0.490** | 0.950 | 0.333 | 0.842 | 3.00 | `e5-small__ventana40__m1__k3__u-__g-` |
| multilingual-e5-base | oracion | sí | 3 | - | - | no | **0.475** | 0.950 | 0.317 | 0.867 | 3.00 | `e5-base__oracion__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 80/20 | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.792 | 3.00 | `minilm__ventana80__m0__k3__u-__g-` |
| multilingual-e5-small | ventana 80/20 | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.842 | 3.00 | `e5-small__ventana80__m0__k3__u-__g-` |
| multilingual-e5-base | ventana 80/20 | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.875 | 3.00 | `e5-base__ventana80__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | oracion | sí | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `minilm__oracion__m1__k3__u-__g-` |
| multilingual-e5-small | ventana 40/10 | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `e5-small__ventana40__m0__k3__u-__g-` |
| multilingual-e5-small | oracion | sí | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `e5-small__oracion__m1__k3__u-__g-` |
| multilingual-e5-base | seccion | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.875 | 3.00 | `e5-base__seccion__m0__k3__u-__g-` |
| multilingual-e5-small | seccion | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.842 | 3.00 | `e5-small__seccion__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | parrafo | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `minilm__parrafo__m0__k3__u-__g-` |
| multilingual-e5-base | ventana 40/10 | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.850 | 3.00 | `e5-base__ventana40__m0__k3__u-__g-` |
| multilingual-e5-base | parrafo | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.850 | 3.00 | `e5-base__parrafo__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | seccion | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.792 | 3.00 | `minilm__seccion__m0__k3__u-__g-` |
| multilingual-e5-small | parrafo | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.842 | 3.00 | `e5-small__parrafo__m0__k3__u-__g-` |
| multilingual-e5-base | oracion | no | 3 | - | - | no | **0.400** | 0.800 | 0.267 | 0.692 | 3.00 | `e5-base__oracion__m0__k3__u-__g-` |
| multilingual-e5-small | oracion | no | 3 | - | - | no | **0.400** | 0.800 | 0.267 | 0.675 | 3.00 | `e5-small__oracion__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | ventana 40/10 | no | 3 | - | - | no | **0.375** | 0.750 | 0.250 | 0.700 | 3.00 | `minilm__ventana40__m0__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 80/20 | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__ventana80__m1__k1__u-__g-` |
| bert-base-multilingual-cased | parrafo | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__parrafo__m1__k1__u-__g-` |
| bert-base-multilingual-cased | seccion | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__seccion__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | oracion | no | 3 | - | - | no | **0.325** | 0.650 | 0.217 | 0.600 | 3.00 | `minilm__oracion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 40/10 | no | 1 | - | - | no | **0.300** | 0.300 | 0.300 | 0.300 | 1.00 | `bert__ventana40__m0__k1__u-__g-` |
| bert-base-multilingual-cased | ventana 80/20 | no | 1 | - | - | no | **0.300** | 0.300 | 0.300 | 0.300 | 1.00 | `bert__ventana80__m0__k1__u-__g-` |
| bert-base-multilingual-cased | oracion | sí | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__oracion__m1__k1__u-__g-` |
| bert-base-multilingual-cased | seccion | no | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__seccion__m0__k1__u-__g-` |
| bert-base-multilingual-cased | parrafo | no | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__parrafo__m0__k1__u-__g-` |
| bert-base-multilingual-cased | ventana 40/10 | sí | 1 | - | - | no | **0.200** | 0.200 | 0.200 | 0.200 | 1.00 | `bert__ventana40__m1__k1__u-__g-` |
| bert-base-multilingual-cased | seccion | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__seccion__m1__k3__u-__g-` |
| bert-base-multilingual-cased | parrafo | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__parrafo__m1__k3__u-__g-` |
| bert-base-multilingual-cased | oracion | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.317 | 3.00 | `bert__oracion__m1__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 80/20 | no | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.342 | 3.00 | `bert__ventana80__m0__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 80/20 | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__ventana80__m1__k3__u-__g-` |
| bert-base-multilingual-cased | parrafo | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.300 | 3.00 | `bert__parrafo__m0__k3__u-__g-` |
| bert-base-multilingual-cased | seccion | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.292 | 3.00 | `bert__seccion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 40/10 | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.317 | 3.00 | `bert__ventana40__m0__k3__u-__g-` |
| bert-base-multilingual-cased | ventana 40/10 | sí | 3 | - | - | no | **0.125** | 0.250 | 0.083 | 0.217 | 3.00 | `bert__ventana40__m1__k3__u-__g-` |
| bert-base-multilingual-cased | oracion | no | 3 | - | - | no | **0.125** | 0.250 | 0.083 | 0.175 | 3.00 | `bert__oracion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | oracion | no | 1 | - | - | no | **0.100** | 0.100 | 0.100 | 0.100 | 1.00 | `bert__oracion__m0__k1__u-__g-` |


## 2. Agente con dos fuentes

_Pendiente._

## 3. Servidor MCP

_Pendiente._

## 4. Atención en NumPy

_Pendiente._

## 5. Bloque de transformer a mano

_Pendiente._

## Costo total en OpenRouter

_Pendiente._
