"""
Modelos
"""

from typing import Literal

from pydantic import BaseModel, Field

# para restringir respuestas
CanalTicket = Literal["Correo", "Página Web", "Whatsapp"]
CategoriaProblema = Literal["Cobros", "Cuenta", "Fraude", "Otro", "Pregunta general", "Técnica"]
TipoCuenta = Literal["Business", "Free", "Premium"]
NivelPrioridad = Literal["Baja", "Media", "Alta", "Critica"]


class PredictionRequest(BaseModel):
    """Campos minimos que necesita el modelo para clasificar un ticket."""

    asunto: str = Field(..., description="Asunto del ticket")
    contenido: str = Field(..., description="Cuerpo/descripción del ticket")
    canal_ticket: CanalTicket
    categoria_problema: CategoriaProblema
    tipo_cuenta: TipoCuenta
    antiguedad_cuenta_dias: int = Field(..., ge=0, description="Antigüedad de la cuenta en días")
    fecha_envio: str = Field(..., description="Fecha de envío del ticket (YYYY-MM-DD)")


class PredictionResponse(BaseModel):
    """Respuesta del endpoint: la prioridad predicha."""

    nivel_prioridad: NivelPrioridad
