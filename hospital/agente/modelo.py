"""El LLM de los agentes: ChatOpenAI de langchain-openai apuntando a OpenRouter."""
import os
from pathlib import Path

MODELO = "deepseek/deepseek-v4-flash-0731"
PRECIO_ENTRADA = 0.04 / 1e6  # USD por token; solo si OpenRouter no informa el costo
PRECIO_SALIDA = 0.64 / 1e6
RAIZ = Path(__file__).resolve().parents[2]


def cargar_env(ruta=RAIZ / ".env"):
    """Carga VARIABLE=valor de .env sin pisar lo que ya está en el entorno."""
    if Path(ruta).exists():
        for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
            if "=" in linea and not linea.lstrip().startswith("#"):
                k, v = linea.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def crear_llm():
    from langchain_openai import ChatOpenAI

    cargar_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("falta OPENROUTER_API_KEY (en el entorno o en .env)")
    return ChatOpenAI(
        model=MODELO,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_retries=3,
        extra_body={"usage": {"include": True}},  # OpenRouter devuelve el costo de cada llamada
    )
