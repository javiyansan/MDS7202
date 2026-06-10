import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar modelo
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


# Definir estructura de entrada
class WaterMeasurement(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Crear app
app = FastAPI()


# GET home
@app.get("/")
def home():
    return {
        "modelo": "XGBoost optimizado con Optuna",
        "problema": "Clasificación binaria de potabilidad del agua",
        "entrada": "9 mediciones químicas: ph, Hardness, Solids , Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity",
        "salida": "potabilidad: 1 (potable) o 0 (no potable)",
    }


# POST predicción
@app.post("/potabilidad/")
def predecir_potabilidad(medicion: WaterMeasurement):
    datos = pd.DataFrame([medicion.model_dump()])
    prediccion = model.predict(datos)[0]
    return {"potabilidad": int(prediccion)}


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
