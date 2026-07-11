"""
definicion de endpoints
"""

from fastapi import FastAPI, HTTPException
from generate_prediction import generate_prediction

from models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="ChaucherApp - Priorización de Tickets",
    description="Clasifica el Nivel_Prioridad de un ticket de soporte al cliente.",
    version="1.0.0",
)


@app.get("/")
def health_check() -> dict:
    """Endpoint simple para verificar que el servicio esta arriba."""
    return {"status": "ok", "servicio": "priorizacion-tickets"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Recibe los datos de un ticket y devuelve su prioridad predicha.

    El tipado estricto de PredictionRequest hace que FastAPI responda 422 si el
    payload no calza. Envolvemos la inferencia en try/except para transformar
    cualquier fallo interno.
    """
    try:
        prioridad = generate_prediction(
            asunto=request.asunto,
            contenido=request.contenido,
            canal_ticket=request.canal_ticket,
            categoria_problema=request.categoria_problema,
            tipo_cuenta=request.tipo_cuenta,
            antiguedad_cuenta_dias=request.antiguedad_cuenta_dias,
            fecha_envio=request.fecha_envio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {e}") from e

    return PredictionResponse(nivel_prioridad=prioridad)
