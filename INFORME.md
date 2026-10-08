# Informe: RAG, MCP y Transformers en el Hospital Arroyo Claro

## 1. RAG vectorial

### Configuración entregada

`hospital/rag/config.json`, que usa `recuperar.py`:

| Encoder | Chunking | Metadatos | Selección | CR dev | Recall | Precision | MRR |
|---|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-base` (`query: ` / `passage: `) | estructural: una sección `##` por fragmento | título > sección antepuestos al embedding | top-2 con margen 0,005 respecto del mejor | **0,967** | 1,000 | 0,950 | 1,000 |

Línea de base obligatoria (BERT multilingüe sin ajustar, promedio de la última capa): **0,350** en su mejor
configuración, igual al recuperador léxico ingenuo del enunciado. La configuración entregada casi triplica ese valor.

### Cómo se eligió

1. **Encoders × chunking × metadatos × k ∈ {1, 3}** para los cuatro encoders (80 corridas). Cada pregunta dev tiene una
   sola frase de evidencia, así que devolver un fragmento extra baja la precisión a la mitad: con k = 3 ningún encoder
   pasa de 0,515. Por eso el ajuste fino se hizo con k chico.
2. **Metadatos:** anteponer "título del documento > sección" al texto del embedding (el fragmento devuelto sigue siendo
   literal) mejora a los cuatro encoders con chunking estructural y k = 1: BERT 0,25 → 0,35, MiniLM 0,70 → 0,90,
   e5-small 0,80 → 0,95, e5-base 0,85 → 1,00. La única combinación donde empeora es BERT con longitud fija de 40 palabras
   (0,30 → 0,20). Muchas secciones no nombran su tema ("La unidad tiene dos franjas de visita…"): el tema está en el encabezado.
3. **Chunking.** Se probaron las tres estrategias de la clase:
   - *Longitud fija con solapamiento*: ventanas de 40 palabras (solapamiento 10) y de 80 (solapamiento 20), sin cruzar
     secciones. Se mide en palabras y no en tokens para que cada fragmento siga siendo texto literal del corpus.
   - *Semántico*: un fragmento por párrafo y, como variante más fina, uno por oración.
   - *Estructural*: un fragmento por sección `##` del Markdown (la introducción del documento es una sección más).

   La estructural y la de longitud fija de 80 palabras empatan arriba. La semántica por oración es la peor para todos
   los encoders: corta el contexto que el encoder necesita para reconocer el tema. Los tamaños de referencia de la
   clase (800–1000 caracteres) no se probaron porque las secciones de este corpus miden 223 caracteres en promedio
   (mediana 193): una ventana así juntaría cuatro o cinco temas distintos en cada fragmento. Se eligió la estructural porque respeta la jerarquía
   del documento y da fragmentos completos (útil para el agente de la parte 2, p. ej. la preparación de la colonoscopía
   entera).
4. **Selección (top-k, umbral, margen):** con e5-base + chunking estructural + metadatos, k = 1 da 1,000 en dev. Pero en dos preguntas
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
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 1 | - | - | no | **1.000** | 1.000 | 1.000 | 1.000 | 1.00 | `e5-base__ventana80__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 1 | - | - | no | **1.000** | 1.000 | 1.000 | 1.000 | 1.00 | `e5-base__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 3 | 0.88 | - | no | **0.983** | 1.000 | 0.975 | 1.000 | 1.05 | `e5-base__seccion__m1__k3__u0.88__g-` |
| multilingual-e5-base | estructural (sección) | sí | 2 | 0.88 | - | no | **0.983** | 1.000 | 0.975 | 1.000 | 1.05 | `e5-base__seccion__m1__k2__u0.88__g-` |
| multilingual-e5-base | estructural (sección) | sí | 3 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__seccion__m1__k3__u-__g0.005` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 2 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__ventana80__m1__k2__u-__g0.005` |
| multilingual-e5-base | estructural (sección) | sí | 2 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__seccion__m1__k2__u-__g0.005` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 3 | - | 0.005 | no | **0.967** | 1.000 | 0.950 | 1.000 | 1.10 | `e5-base__ventana80__m1__k3__u-__g0.005` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 2 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__ventana80__m1__k2__u-__g0.01` |
| multilingual-e5-base | estructural (sección) | sí | 3 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__seccion__m1__k3__u-__g0.01` |
| multilingual-e5-small | estructural (sección) | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-small__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | semántico (párrafo) | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-base__parrafo__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 2 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__seccion__m1__k2__u-__g0.01` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 3 | - | 0.01 | no | **0.950** | 1.000 | 0.925 | 1.000 | 1.15 | `e5-base__ventana80__m1__k3__u-__g0.01` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (80 palabras, solap. 20) | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `minilm__ventana80__m1__k1__u-__g-` |
| multilingual-e5-small | longitud fija (80 palabras, solap. 20) | sí | 1 | - | - | no | **0.950** | 0.950 | 0.950 | 0.950 | 1.00 | `e5-small__ventana80__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (párrafo) | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `minilm__parrafo__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 2 | 0.85 | - | no | **0.900** | 1.000 | 0.850 | 1.000 | 1.30 | `e5-base__seccion__m1__k2__u0.85__g-` |
| multilingual-e5-small | semántico (párrafo) | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `e5-small__parrafo__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | estructural (sección) | sí | 1 | - | - | no | **0.900** | 0.900 | 0.900 | 0.900 | 1.00 | `minilm__seccion__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 2 | - | 0.02 | no | **0.900** | 1.000 | 0.850 | 1.000 | 1.30 | `e5-base__seccion__m1__k2__u-__g0.02` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 2 | - | 0.02 | no | **0.883** | 1.000 | 0.825 | 1.000 | 1.35 | `e5-base__ventana80__m1__k2__u-__g0.02` |
| multilingual-e5-base | estructural (sección) | sí | 3 | 0.85 | - | no | **0.867** | 1.000 | 0.817 | 1.000 | 1.50 | `e5-base__seccion__m1__k3__u0.85__g-` |
| multilingual-e5-base | estructural (sección) | sí | 3 | - | 0.02 | no | **0.867** | 1.000 | 0.817 | 1.000 | 1.50 | `e5-base__seccion__m1__k3__u-__g0.02` |
| multilingual-e5-base | longitud fija (40 palabras, solap. 10) | sí | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__ventana40__m1__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | no | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__seccion__m0__k1__u-__g-` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | no | 1 | - | - | no | **0.850** | 0.850 | 0.850 | 0.850 | 1.00 | `e5-base__ventana80__m0__k1__u-__g-` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 3 | - | 0.02 | no | **0.850** | 1.000 | 0.792 | 1.000 | 1.55 | `e5-base__ventana80__m1__k3__u-__g0.02` |
| multilingual-e5-base | semántico (párrafo) | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | semántico (párrafo) | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | estructural (sección) | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__seccion__m0__k1__u-__g-` |
| multilingual-e5-base | longitud fija (40 palabras, solap. 10) | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__ventana40__m0__k1__u-__g-` |
| multilingual-e5-base | semántico (oración) | sí | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-base__oracion__m1__k1__u-__g-` |
| multilingual-e5-small | longitud fija (80 palabras, solap. 20) | no | 1 | - | - | no | **0.800** | 0.800 | 0.800 | 0.800 | 1.00 | `e5-small__ventana80__m0__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 2 | 0.82 | - | no | **0.750** | 1.000 | 0.625 | 1.000 | 1.75 | `e5-base__seccion__m1__k2__u0.82__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (párrafo) | no | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__parrafo__m0__k1__u-__g-` |
| multilingual-e5-small | longitud fija (40 palabras, solap. 10) | no | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__ventana40__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (40 palabras, solap. 10) | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__ventana40__m1__k1__u-__g-` |
| multilingual-e5-small | semántico (oración) | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__oracion__m1__k1__u-__g-` |
| multilingual-e5-small | longitud fija (40 palabras, solap. 10) | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `e5-small__ventana40__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (oración) | sí | 1 | - | - | no | **0.750** | 0.750 | 0.750 | 0.750 | 1.00 | `minilm__oracion__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | estructural (sección) | no | 1 | - | - | no | **0.700** | 0.700 | 0.700 | 0.700 | 1.00 | `minilm__seccion__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (80 palabras, solap. 20) | no | 1 | - | - | no | **0.700** | 0.700 | 0.700 | 0.700 | 1.00 | `minilm__ventana80__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (40 palabras, solap. 10) | no | 1 | - | - | no | **0.650** | 0.650 | 0.650 | 0.650 | 1.00 | `minilm__ventana40__m0__k1__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 3 | 0.82 | - | no | **0.633** | 1.000 | 0.508 | 1.000 | 2.45 | `e5-base__seccion__m1__k3__u0.82__g-` |
| multilingual-e5-base | semántico (oración) | no | 1 | - | - | no | **0.600** | 0.600 | 0.600 | 0.600 | 1.00 | `e5-base__oracion__m0__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (oración) | no | 1 | - | - | no | **0.550** | 0.550 | 0.550 | 0.550 | 1.00 | `minilm__oracion__m0__k1__u-__g-` |
| multilingual-e5-small | semántico (oración) | no | 1 | - | - | no | **0.550** | 0.550 | 0.550 | 0.550 | 1.00 | `e5-small__oracion__m0__k1__u-__g-` |
| multilingual-e5-small | longitud fija (80 palabras, solap. 20) | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.967 | 3.00 | `e5-small__ventana80__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (80 palabras, solap. 20) | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.975 | 3.00 | `minilm__ventana80__m1__k3__u-__g-` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 1.000 | 3.00 | `e5-base__ventana80__m1__k3__u-__g-` |
| multilingual-e5-base | longitud fija (40 palabras, solap. 10) | sí | 3 | - | - | no | **0.515** | 1.000 | 0.350 | 0.925 | 3.00 | `e5-base__ventana40__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (40 palabras, solap. 10) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.875 | 3.00 | `minilm__ventana40__m1__k3__u-__g-` |
| multilingual-e5-base | semántico (párrafo) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.967 | 3.00 | `e5-base__parrafo__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | estructural (sección) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.950 | 3.00 | `minilm__seccion__m1__k3__u-__g-` |
| multilingual-e5-small | semántico (párrafo) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.942 | 3.00 | `e5-small__parrafo__m1__k3__u-__g-` |
| multilingual-e5-base | estructural (sección) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 1.000 | 3.00 | `e5-base__seccion__m1__k3__u-__g-` |
| multilingual-e5-small | estructural (sección) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.967 | 3.00 | `e5-small__seccion__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (párrafo) | sí | 3 | - | - | no | **0.500** | 1.000 | 0.333 | 0.950 | 3.00 | `minilm__parrafo__m1__k3__u-__g-` |
| multilingual-e5-small | longitud fija (40 palabras, solap. 10) | sí | 3 | - | - | no | **0.490** | 0.950 | 0.333 | 0.842 | 3.00 | `e5-small__ventana40__m1__k3__u-__g-` |
| multilingual-e5-base | semántico (oración) | sí | 3 | - | - | no | **0.475** | 0.950 | 0.317 | 0.867 | 3.00 | `e5-base__oracion__m1__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (80 palabras, solap. 20) | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.792 | 3.00 | `minilm__ventana80__m0__k3__u-__g-` |
| multilingual-e5-small | longitud fija (80 palabras, solap. 20) | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.842 | 3.00 | `e5-small__ventana80__m0__k3__u-__g-` |
| multilingual-e5-base | longitud fija (80 palabras, solap. 20) | no | 3 | - | - | no | **0.465** | 0.900 | 0.317 | 0.875 | 3.00 | `e5-base__ventana80__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (oración) | sí | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `minilm__oracion__m1__k3__u-__g-` |
| multilingual-e5-small | longitud fija (40 palabras, solap. 10) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `e5-small__ventana40__m0__k3__u-__g-` |
| multilingual-e5-small | semántico (oración) | sí | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `e5-small__oracion__m1__k3__u-__g-` |
| multilingual-e5-base | estructural (sección) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.875 | 3.00 | `e5-base__seccion__m0__k3__u-__g-` |
| multilingual-e5-small | estructural (sección) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.842 | 3.00 | `e5-small__seccion__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (párrafo) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.817 | 3.00 | `minilm__parrafo__m0__k3__u-__g-` |
| multilingual-e5-base | longitud fija (40 palabras, solap. 10) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.850 | 3.00 | `e5-base__ventana40__m0__k3__u-__g-` |
| multilingual-e5-base | semántico (párrafo) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.850 | 3.00 | `e5-base__parrafo__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | estructural (sección) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.792 | 3.00 | `minilm__seccion__m0__k3__u-__g-` |
| multilingual-e5-small | semántico (párrafo) | no | 3 | - | - | no | **0.450** | 0.900 | 0.300 | 0.842 | 3.00 | `e5-small__parrafo__m0__k3__u-__g-` |
| multilingual-e5-base | semántico (oración) | no | 3 | - | - | no | **0.400** | 0.800 | 0.267 | 0.692 | 3.00 | `e5-base__oracion__m0__k3__u-__g-` |
| multilingual-e5-small | semántico (oración) | no | 3 | - | - | no | **0.400** | 0.800 | 0.267 | 0.675 | 3.00 | `e5-small__oracion__m0__k3__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | longitud fija (40 palabras, solap. 10) | no | 3 | - | - | no | **0.375** | 0.750 | 0.250 | 0.700 | 3.00 | `minilm__ventana40__m0__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (80 palabras, solap. 20) | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__ventana80__m1__k1__u-__g-` |
| bert-base-multilingual-cased | semántico (párrafo) | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__parrafo__m1__k1__u-__g-` |
| bert-base-multilingual-cased | estructural (sección) | sí | 1 | - | - | no | **0.350** | 0.350 | 0.350 | 0.350 | 1.00 | `bert__seccion__m1__k1__u-__g-` |
| paraphrase-multilingual-MiniLM-L12-v2 | semántico (oración) | no | 3 | - | - | no | **0.325** | 0.650 | 0.217 | 0.600 | 3.00 | `minilm__oracion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (40 palabras, solap. 10) | no | 1 | - | - | no | **0.300** | 0.300 | 0.300 | 0.300 | 1.00 | `bert__ventana40__m0__k1__u-__g-` |
| bert-base-multilingual-cased | longitud fija (80 palabras, solap. 20) | no | 1 | - | - | no | **0.300** | 0.300 | 0.300 | 0.300 | 1.00 | `bert__ventana80__m0__k1__u-__g-` |
| bert-base-multilingual-cased | semántico (oración) | sí | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__oracion__m1__k1__u-__g-` |
| bert-base-multilingual-cased | estructural (sección) | no | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__seccion__m0__k1__u-__g-` |
| bert-base-multilingual-cased | semántico (párrafo) | no | 1 | - | - | no | **0.250** | 0.250 | 0.250 | 0.250 | 1.00 | `bert__parrafo__m0__k1__u-__g-` |
| bert-base-multilingual-cased | longitud fija (40 palabras, solap. 10) | sí | 1 | - | - | no | **0.200** | 0.200 | 0.200 | 0.200 | 1.00 | `bert__ventana40__m1__k1__u-__g-` |
| bert-base-multilingual-cased | estructural (sección) | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__seccion__m1__k3__u-__g-` |
| bert-base-multilingual-cased | semántico (párrafo) | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__parrafo__m1__k3__u-__g-` |
| bert-base-multilingual-cased | semántico (oración) | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.317 | 3.00 | `bert__oracion__m1__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (80 palabras, solap. 20) | no | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.342 | 3.00 | `bert__ventana80__m0__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (80 palabras, solap. 20) | sí | 3 | - | - | no | **0.200** | 0.400 | 0.133 | 0.367 | 3.00 | `bert__ventana80__m1__k3__u-__g-` |
| bert-base-multilingual-cased | semántico (párrafo) | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.300 | 3.00 | `bert__parrafo__m0__k3__u-__g-` |
| bert-base-multilingual-cased | estructural (sección) | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.292 | 3.00 | `bert__seccion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (40 palabras, solap. 10) | no | 3 | - | - | no | **0.175** | 0.350 | 0.117 | 0.317 | 3.00 | `bert__ventana40__m0__k3__u-__g-` |
| bert-base-multilingual-cased | longitud fija (40 palabras, solap. 10) | sí | 3 | - | - | no | **0.125** | 0.250 | 0.083 | 0.217 | 3.00 | `bert__ventana40__m1__k3__u-__g-` |
| bert-base-multilingual-cased | semántico (oración) | no | 3 | - | - | no | **0.125** | 0.250 | 0.083 | 0.175 | 3.00 | `bert__oracion__m0__k3__u-__g-` |
| bert-base-multilingual-cased | semántico (oración) | no | 1 | - | - | no | **0.100** | 0.100 | 0.100 | 0.100 | 1.00 | `bert__oracion__m0__k1__u-__g-` |

## 2. Agente con dos fuentes

### Cómo está hecho

- **LangChain, sin LangGraph.** `ChatOpenAI` de `langchain-openai` apuntando a OpenRouter
  (`deepseek/deepseek-v4-flash-0731`, temperatura 0) y seis tools de LangChain (`hospital/herramientas.py`) con los
  nombres que pide el evaluador. El tool calling es un bucle explícito (`hospital/agente/bucle.py`). El modelo, con las
  tools enlazadas por `bind_tools`, pide herramientas. Se ejecutan y sus resultados vuelven como `ToolMessage`, hasta que
  el modelo responde sin pedir nada (máximo 6 llamadas). Si una herramienta no existe o falla, el error vuelve al modelo
  como texto para que se corrija.
- **Fuentes.** `buscar_documentos` usa el recuperador de la parte 1 y antepone a cada fragmento su origen
  (`[Documento > Sección]`), así el modelo sabe de dónde sale cada dato. Las otras cinco llaman a la API
  (`hospital/api_cliente.py`) y devuelven el JSON tal cual, también cuando es un error con la lista de opciones válidas.
- **Descripciones** (`hospital/descripciones.py`). Cada una dice qué fuente consulta, para qué preguntas sirve, qué
  NO tiene (por ejemplo, `consultar_turnos` aclara que lo que hay que llevar está en los documentos) y los valores
  válidos del parámetro. Son las mismas que va a publicar el servidor MCP de la parte 3.
- **Prompt de sistema** (`bucle.py`). Toda la información sale de las herramientas. Normas y procedimientos van a los
  documentos, el estado de hoy va a la API, y cada parte de una pregunta mixta se cubre con su fuente. La respuesta usa
  solo datos de los resultados y, si algo no está, lo dice.
- **Logs** (`hospital/agente/corrida.py`). Cada corrida deja `experimentos/corridas/agente_<fecha>.md` con, por pregunta,
  cada llamada al modelo (tokens de entrada, de cache, de salida y de razonamiento, costo en USD que informa OpenRouter
  e id de generación), cada herramienta con sus argumentos y su resultado completo, y la respuesta final. Arriba
  resume totales por pregunta y de la corrida.

### Corridas

Cada corrida tiene en `experimentos/corridas/` su log `.md`, su `respuestas.jsonl` y su `.eval.json`. La entregada es
la 4 (`respuestas.jsonl` en la raíz = `agente_20260929-105719`).

| # | Corrida | Cambio respecto de la anterior | Ruteo | CR | F | AR | Costo agente (USD) | Costo juez (USD) |
|---|---|---|---|---|---|---|---|---|
| 1 | `agente_20260929-104728` | versión inicial; `buscar_documentos` con la selección de la parte 1 (top-2, margen 0,005) | 1,00 | 4,917 | 5,00 | 4,917 | 0,001405 | 0,01667 |
| 2 | `agente_20260929-105117` | `buscar_documentos` devuelve siempre 3 fragmentos | 1,00 | 4,417 | 5,00 | 5,000 | 0,001464 | 0,02141 |
| 3 | `agente_20260929-105408` | hasta 3 fragmentos, solo los que están a menos de 0,02 del mejor | 1,00 | 4,833 | 5,00 | 4,917 | 0,001439 | 0,01797 |
| 4 | `agente_20260929-105719` | vuelve la selección de la parte 1; la descripción de `buscar_documentos` pide usar las palabras del paciente y una búsqueda por tema | **1,00** | **5,000** | **5,00** | **5,000** | **0,001458** | 0,01744 |

Cada corrida hace 24 llamadas al modelo (2 por pregunta: una que pide las herramientas y otra que responde) y usa unos
40 mil tokens de entrada y 2,3 mil de salida. En las preguntas mixtas (A10–A12) el modelo pide las dos herramientas
en paralelo en la misma llamada. El costo del agente es de alrededor de USD 0,0015 por corrida, y el juez cuesta más de
diez veces eso.

### Dónde falló el agente y qué se cambió

**A11** ("Necesito turno con cardiología, ¿cuál es el primero y qué tengo que llevar?") fue la única pregunta con
problemas en la corrida 1 (CR 4, AR 4). El log muestra que el ruteo estuvo bien: llamó a `consultar_turnos` y a
`buscar_documentos`. Pero buscó *"requisitos y documentación para turno en consultorios externos de cardiología"*, y el
recuperador devolvió solo la sección "Cómo se pide un turno", que menciona la derivación. La sección "Qué llevar a la
primera consulta" (DNI, credencial, derivación, estudios previos) quedó tercera, a 0,018 del primero y fuera del
margen de 0,005. La respuesta fue fiel a lo que recibió (F 5), pero incompleta.

Primero se probó darle más fragmentos al agente, porque la métrica de la parte 1 castiga cada fragmento extra y la del
juez no tanto:

- Con **top-3 fijo** (corrida 2), A11 sube a 5, pero el ruido baja el CR de A01–A04 y A12 (3 y 4): el juez penaliza
  los fragmentos que no hacen falta. El promedio de CR cae a 4,417.
- Con **margen 0,02** (corrida 3), el resto vuelve a 5. Pero A11 falla otra vez (CR 3), porque el modelo redactó la
  consulta de otra forma y la sección correcta ya no quedaba cerca del primero.

El problema no era la cantidad de fragmentos sino la consulta. Buscar "requisitos y documentación" no se parece al
encabezado "Qué llevar a la primera consulta"; buscar con las palabras del paciente ("qué tengo que llevar") sí. Por
eso en la corrida 4 se volvió a la selección de la parte 1 y se cambió la descripción de `buscar_documentos`: la
consulta va "con las mismas palabras que usó el paciente" y, si la pregunta toca varios temas, se hace una búsqueda por
tema. En A11 el modelo buscó *"qué llevar a un turno de cardiología…"*, recibió la sección correcta y contestó completo.

En el mismo cambio se reemplazó el ejemplo de la descripción ("horario de visita de los abuelos en neonatología"), que
era literalmente la pregunta A01 de dev, por uno que no aparece en ninguna pregunta dev ("cómo es una consulta por
telemedicina"). Así se evita ajustar el agente a las preguntas dev.

**Límites de esta medición.** Son 12 preguntas y una corrida por configuración. El modelo redacta la consulta distinto
en cada corrida aun con temperatura 0: en A11, con la misma descripción, las corridas 1 a 3 buscaron "…consultorios
externos de cardiología", "…consultorios externos cardiología" y "…consultorios externos" a secas. Así que una
diferencia de una pregunta entre corridas está dentro del ruido. Lo que se sostiene en las cuatro corridas es el ruteo
(1,00 siempre) y la fidelidad (5,00 siempre): el agente no inventó datos en ninguna respuesta.

## 3. Servidor MCP

### Implementación y verificación

Las seis herramientas de la parte 2 se exponen desde `servidor_mcp.py` con el SDK oficial `mcp` y transporte stdio. Sus
funciones delegan a los mismos módulos de documentos y API; las descripciones compartidas viven en
`hospital/descripciones.py`. `agente_mcp.py` descubre las tools con `tools/list`, las convierte a tools de LangChain con
`langchain-mcp-adapters` y sus invocaciones usan `tools/call`. La sesión MCP queda abierta durante toda la corrida para
evitar reiniciar el servidor y recargar el encoder por cada llamada.

Se verificó el descubrimiento de las seis herramientas y una llamada a cada una. Las cinco herramientas de la API
devolvieron JSON del hospital y `buscar_documentos` devolvió fragmentos del corpus. MCP Inspector se conectó por stdio
al servidor. El registro de la corrida está en
`experimentos/corridas/agente_mcp_20261001-200728.md`; las respuestas y evaluación están en `respuestas_mcp.jsonl` y
`respuestas_mcp.jsonl.eval.json`. El Inspector se conectó, listó las seis herramientas y permitió invocar cada una
con éxito. Las siete capturas PNG (listado y seis resultados) están en `experimentos/inspector/`.
Para apuntar a la API local en el puerto alternativo, se inició Inspector con
`npx @modelcontextprotocol/inspector -e HOSPITAL_API=http://127.0.0.1:18765 .venv/bin/python servidor_mcp.py`;
la opción `-e` pasa explícitamente esa variable al proceso MCP.

### Comparación con la parte 2

| Agente | Ruteo | Relevancia del contexto | Fidelidad | Pertinencia de respuesta | Costo agente (USD) | Costo juez (USD) |
|---|---:|---:|---:|---:|---:|---:|
| Parte 2 (`agente_20260929-105719`) | 1,000 | 5,000 | 5,000 | 5,000 | 0,001458 | 0,01744 |
| Parte 3 (`agente_mcp_20261001-200728`) | 1,000 | 4,833 | 5,000 | 4,917 | 0,003524 | 0,02002 |

Las dos corridas enrutan correctamente las 12 preguntas y mantienen fidelidad perfecta. La diferencia está en A11:
la parte 2 recuperó «Qué llevar a la primera consulta» con DNI, credencial, derivación y estudios previos. En la corrida
MCP, la primera búsqueda devolvió «Llegada» y la segunda una sección sobre cómo pedir el turno; faltó la lista de
documentación. El juez bajó la relevancia del contexto a 3 y la pertinencia de la respuesta a 4 para esa pregunta.
Esto coincide con los logs: el agente MCP hizo una llamada adicional al modelo en A11 (25 en total frente a 24 en la
parte 2) y entregó contexto menos pertinente. Por eso sus costos de agente y juez también fueron algo mayores; una
corrida por versión no alcanza para atribuir la diferencia al transporte MCP por sí solo.

## 4. Atención en NumPy

La implementaciÃ³n estÃ¡ en `atencion.py` y usa Ãºnicamente NumPy. Se implementaron las
cinco operaciones pedidas:

- `softmax(M)`: calcula el softmax sobre el Ãºltimo eje. Antes de aplicar la exponencial
  resta el mÃ¡ximo de cada fila, de modo que tambiÃ©n sea estable con valores grandes.
- `atencion(Q, K, V, mascara=False)`: calcula los puntajes `Q K^T / sqrt(d_k)`, aplica
  opcionalmente la mÃ¡scara causal poniendo `-inf` sobre la diagonal superior, y
  devuelve tanto `A V` como la matriz de pesos `A`.
- `autoatencion(X, Wq, Wk, Wv, mascara=False)`: proyecta la entrada en consultas,
  claves y valores, y reutiliza la atenciÃ³n anterior.
- `multicabeza(X, cabezas, Wo, mascara=False)`: ejecuta cada cabeza, concatena sus
  salidas y aplica la proyecciÃ³n final `Wo`.
- `layer_norm(x, eps=1e-5)`: normaliza cada fila a media aproximadamente cero y
  varianza aproximadamente uno, sin parÃ¡metros aprendidos `gamma` ni `beta`.

La verificaciÃ³n se hizo con:

```bash
python3 atencion/test_atencion.py atencion.py
```

El resultado fue **14 tests ejecutados, 14 aprobados (`OK`)**. Se comprobaron el
softmax estable, la escala por `sqrt(d_k)`, la mÃ¡scara causal, las formas de las
salidas, la permutaciÃ³n de filas, la concatenaciÃ³n de mÃºltiples cabezas y las
invariantes de `layer_norm`.

## 5. Bloque de transformer a mano

_Pendiente._

## Costo total en OpenRouter

En las corridas entregadas de las partes 2 y 3, el costo medido es **USD 0,042442**: USD 0,004982 del agente y USD
0,03746 del juez. La parte 3 costó USD 0,003524 para las llamadas a DeepSeek y USD 0,02002 para Gemini como juez.
Las partes 1, 4 y 5 no usan OpenRouter en sus procedimientos de evaluación.
