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

_Pendiente._

## 3. Servidor MCP

_Pendiente._

## 4. Atención en NumPy

_Pendiente._

## 5. Bloque de transformer a mano

_Pendiente._

## Costo total en OpenRouter

_Pendiente._
