import json
import subprocess
import sys
import time
import urllib.request

import pytest

from hospital import api_cliente

PUERTO = 8799


@pytest.fixture(scope="module", autouse=True)
def api(monkeypatch_module):
    proc = subprocess.Popen([sys.executable, "api/servidor.py", "--puerto", str(PUERTO)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{PUERTO}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{url}/espera", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    monkeypatch_module.setenv("HOSPITAL_API", url)
    yield
    proc.terminate()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_camas_devuelve_json_con_libres():
    d = json.loads(api_cliente.camas("pediatria"))
    assert d["datos"]["libres"] == 7


def test_nombres_con_tildes_y_espacios():
    d = json.loads(api_cliente.camas("Terapia Intensiva"))
    assert d["sector"] == "terapia_intensiva"


def test_error_trae_las_opciones_validas():
    d = json.loads(api_cliente.camas("oncologia"))
    assert "error" in d and "pediatria" in d["opciones"]


def test_guardia_turnos_farmacia_espera():
    assert json.loads(api_cliente.guardia("cardiologia"))["datos"]
    assert json.loads(api_cliente.turnos("traumatologia"))["datos"]
    assert "enalapril" in json.loads(api_cliente.farmacia("enalapril 10 mg"))["medicamento"]
    assert "verde" in json.loads(api_cliente.espera())["minutos_por_nivel"]


def test_api_caida_devuelve_error_legible(monkeypatch):
    monkeypatch.setenv("HOSPITAL_API", "http://localhost:1")
    assert "error" in json.loads(api_cliente.espera())
