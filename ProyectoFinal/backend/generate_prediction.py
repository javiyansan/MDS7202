"""
Adaptación del pipeline.

"""

import os
from pathlib import Path

import cloudpickle
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# GOOGLE_API_KEY desde el .env
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "modelo_final.pkl"))
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 1024

# Nombres de columna de los embeddings
EMBEDDING_COLS = [f"embedding_dim_{i}" for i in range(1, EMBEDDING_DIM + 1)]

with open(MODEL_PATH, "rb") as f:
    _PIPELINE = cloudpickle.load(f)

# Instanciamos el cliente de embeddings una vez y lo reutilizamos
_embedder = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    output_dimensionality=EMBEDDING_DIM,
)


def _contar_caracteres(asunto: str, contenido: str) -> int:
    """Reconstruye N_Caracteres_Ticket tal como venía en los datos originales.

    Deducido explorando tickets.parquet
    N_Caracteres_Ticket es la suma de los largos del asunto y del contenido,
    No se cuentan los '\\r'.
    """
    asunto_norm = asunto.replace("\r\n", "\n")
    contenido_norm = contenido.replace("\r\n", "\n")
    return len(asunto_norm) + len(contenido_norm)


def _vectorizar_texto(asunto: str, contenido: str) -> list[float]:
    """Genera el embedding del ticket igual al usado en entrenamiento.

    El formato de concatenación debe ser igual el que se uso para calcular
    los embeddings entregados, si no el vector no vive en el mismo espacio.
    """
    texto_para_embedding = f"Asunto_Ticket: {asunto}\nContenido_Ticket: {contenido}\n"
    return _embedder.embed_query(texto_para_embedding)


def generate_prediction(
    asunto: str,
    contenido: str,
    canal_ticket: str,
    categoria_problema: str,
    tipo_cuenta: str,
    antiguedad_cuenta_dias: int,
    fecha_envio: str,
) -> str:
    """Recibe los campos minimos de un ticket y devuelve su Nivel_Prioridad.

    Pasos:
      1. Derivar las features que el usuario no entrega (N_Caracteres, Texto).
      2. Vectorizar el texto con gemini-embedding-001 (misma dim y formato).
      3. Armar un DataFrame de 1 fila con TODAS las columnas que espera el pipeline.
      4. Predecir con el pipeline cargado y devolver la etiqueta.
    """
    # 1. Features derivadas
    n_caracteres = _contar_caracteres(asunto, contenido)
    texto_bow = f"{asunto} {contenido}"

    # 2. Embedding del texto
    vector = _vectorizar_texto(asunto, contenido)
    embedding_dict = dict(zip(EMBEDDING_COLS, vector, strict=False))

    # 3. Construccion del df de entrada (1 fila) con el esquema de X_full
    fila = {
        "N_Caracteres_Ticket": n_caracteres,
        "Canal_Ticket": canal_ticket,
        "Categoría_Problema": categoria_problema,
        "Usuario-Tipo_de_Cuenta": tipo_cuenta,
        "Usuario-Antiguedad_Cuenta_Dias": antiguedad_cuenta_dias,
        "Fecha_Envío": fecha_envio,
        "Texto": texto_bow,
        **embedding_dict,
    }
    X = pd.DataFrame([fila])

    # 4. Predicción
    prediccion = _PIPELINE.predict(X)[0]
    return str(prediccion)


if __name__ == "__main__":
    # Ejecucion de prueba con datos de muestra
    ejemplo = generate_prediction(
        asunto="Cobro desconocido en mi tarjeta digital",
        contenido=(
            "Hola, buenas. Me apareció un cobro que no reconozco en mi tarjeta "
            "y necesito que lo revisen con urgencia porque no fui yo."
        ),
        canal_ticket="Whatsapp",
        categoria_problema="Fraude",
        tipo_cuenta="Free",
        antiguedad_cuenta_dias=314,
        fecha_envio="2024-01-02",
    )
    print(f"Nivel de prioridad predicho: {ejemplo}")
