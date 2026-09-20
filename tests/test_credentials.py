"""Carga de credenciales desde ``.env`` sin instalar nada a nivel de usuario.

El token de Kaggle es un secreto. Las propiedades que importan aqui son
defensivas: que una variable ya definida en la shell gane sobre el archivo, que
los valores nunca salgan en el diccionario de retorno (acaba en logs), y que el
estado reportado diga si hay credencial sin revelar cual.
"""

from __future__ import annotations

import os

import pytest

from fraud_adaptive.data import kaggle_credentials_status, load_dotenv


@pytest.fixture
def entorno_limpio(monkeypatch):
    """Aisla las variables de Kaggle: la maquina de quien corre puede tenerlas."""
    for clave in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "OTRA_VAR"):
        monkeypatch.delenv(clave, raising=False)
    return monkeypatch


def test_carga_una_variable(tmp_path, entorno_limpio):
    archivo = tmp_path / ".env"
    archivo.write_text("KAGGLE_API_TOKEN=KGAT-abc123\n", encoding="utf-8")

    cargadas = load_dotenv(archivo)
    assert os.environ["KAGGLE_API_TOKEN"] == "KGAT-abc123"
    assert list(cargadas) == ["KAGGLE_API_TOKEN"]


def test_no_devuelve_los_valores(tmp_path, entorno_limpio):
    """El diccionario de retorno acaba en mensajes: no puede llevar el secreto."""
    archivo = tmp_path / ".env"
    archivo.write_text("KAGGLE_API_TOKEN=secreto-que-no-debe-salir\n", encoding="utf-8")

    cargadas = load_dotenv(archivo)
    assert "secreto-que-no-debe-salir" not in str(cargadas)
    assert cargadas["KAGGLE_API_TOKEN"] == "<definida>"


def test_una_variable_del_entorno_gana_sobre_el_archivo(tmp_path, entorno_limpio):
    """Exportar en la shell es mas explicito que un archivo; debe prevalecer."""
    entorno_limpio.setenv("KAGGLE_API_TOKEN", "de-la-shell")
    archivo = tmp_path / ".env"
    archivo.write_text("KAGGLE_API_TOKEN=del-archivo\n", encoding="utf-8")

    load_dotenv(archivo)
    assert os.environ["KAGGLE_API_TOKEN"] == "de-la-shell"


def test_override_si_se_pide_explicitamente(tmp_path, entorno_limpio):
    entorno_limpio.setenv("KAGGLE_API_TOKEN", "de-la-shell")
    archivo = tmp_path / ".env"
    archivo.write_text("KAGGLE_API_TOKEN=del-archivo\n", encoding="utf-8")

    load_dotenv(archivo, override=True)
    assert os.environ["KAGGLE_API_TOKEN"] == "del-archivo"


def test_ignora_comentarios_lineas_vacias_y_basura(tmp_path, entorno_limpio):
    archivo = tmp_path / ".env"
    archivo.write_text(
        "# un comentario\n"
        "\n"
        "   \n"
        "linea sin igual\n"
        "OTRA_VAR=valor\n"
        "SIN_VALOR=\n",
        encoding="utf-8",
    )
    cargadas = load_dotenv(archivo)
    assert list(cargadas) == ["OTRA_VAR"]
    assert os.environ["OTRA_VAR"] == "valor"
    assert "SIN_VALOR" not in os.environ


def test_quita_comillas_alrededor_del_valor(tmp_path, entorno_limpio):
    """Un token entre comillas no debe llegar con las comillas al cliente."""
    archivo = tmp_path / ".env"
    archivo.write_text('KAGGLE_API_TOKEN="KGAT-abc"\nOTRA_VAR=\'x\'\n', encoding="utf-8")
    load_dotenv(archivo)
    assert os.environ["KAGGLE_API_TOKEN"] == "KGAT-abc"
    assert os.environ["OTRA_VAR"] == "x"


def test_archivo_ausente_no_falla(tmp_path, entorno_limpio):
    """Sin .env el pipeline debe seguir: solo la descarga necesita credenciales."""
    assert load_dotenv(tmp_path / "no_existe.env") == {}


def test_el_estado_no_revela_credenciales(entorno_limpio):
    entorno_limpio.setenv("KAGGLE_API_TOKEN", "KGAT-supersecreto")
    estado = kaggle_credentials_status()
    assert estado["KAGGLE_API_TOKEN"] is True
    assert "supersecreto" not in str(estado)
    # Todos los valores son booleanos: no hay forma de filtrar el secreto.
    assert all(isinstance(v, bool) for v in estado.values())


def test_el_estado_detecta_ausencia(entorno_limpio):
    estado = kaggle_credentials_status()
    assert estado["KAGGLE_API_TOKEN"] is False
    assert estado["KAGGLE_USERNAME_y_KEY"] is False


def test_username_y_key_requieren_ambos(entorno_limpio):
    entorno_limpio.setenv("KAGGLE_USERNAME", "alguien")
    assert kaggle_credentials_status()["KAGGLE_USERNAME_y_KEY"] is False
    entorno_limpio.setenv("KAGGLE_KEY", "clave")
    assert kaggle_credentials_status()["KAGGLE_USERNAME_y_KEY"] is True
