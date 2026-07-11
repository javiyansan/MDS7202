"""Capa de servicios del frontend"""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def enviar_prediccion(payload: dict) -> dict:
    """Envia el payload al endpoint /predict y devuelve la respuesta como dict.

    Lanza una excepcion si el backend responde con un código de error, para que
    la capa de UI pueda mostrar un mensaje adecuado.
    """
    respuesta = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=60)
    respuesta.raise_for_status()
    return respuesta.json()
