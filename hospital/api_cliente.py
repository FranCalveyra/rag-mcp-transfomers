"""Cliente de la API del hospital (api/servidor.py). Sin LangChain: lo usan las tools de la parte 2 y el
servidor MCP de la parte 3.

Cada funcion devuelve el cuerpo JSON como texto, tambien cuando la API responde con error (404 con la lista
de opciones validas), para que el modelo pueda corregirse y reintentar.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def _get(ruta, **params):
    base = os.environ.get("HOSPITAL_API", "http://localhost:8765")
    url = f"{base}{ruta}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:  # 400/404 traen JSON con "error" y "opciones"
        return e.read().decode("utf-8")
    except OSError as e:
        return json.dumps({"error": f"la API del hospital no responde en {base}: {e}"}, ensure_ascii=False)


def camas(sector):
    return _get("/camas", sector=sector)


def guardia(especialidad):
    return _get("/guardia", especialidad=especialidad)


def turnos(especialidad):
    return _get("/turnos", especialidad=especialidad)


def farmacia(medicamento):
    return _get("/farmacia", medicamento=medicamento)


def espera():
    return _get("/espera")
