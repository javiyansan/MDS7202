import os
import pickle

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("water_potability.csv")
df = df.fillna(df.median(numeric_only=True))  # imputar nulos con mediana

X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def optimize_model():
    # Optimización de hiperparámetros del modelo con optuna y mlflow
    # -----------------------------------------------------------------
    # Nombre reconocible para experimento
    experiment = mlflow.get_experiment_by_name("Potabilidad_XGBoost_experimento2")
    if experiment is None:
        experiment_id = mlflow.create_experiment("Potabilidad_XGBoost_experimento2")
    else:
        experiment_id = experiment.experiment_id

    def objective_function(trial):
        # Definición de hiperparámetros
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }

        # Nombre interpretable para el run
        run_name = f"XGBoost con lr {params['learning_rate']:.3f} y depth {params['max_depth']}"

        # Entrenamiento de XGBoost (mlflow)
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            model = XGBClassifier(seed=42, eval_metric="logloss", **params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
            )
            # Prediccion y evaluacion
            yhat = model.predict(X_valid)
            valid_f1 = f1_score(y_valid, yhat)

            # Registrar resultados en mlfloww
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", valid_f1)
            mlflow.sklearn.log_model(model, name="model")

        return valid_f1

    # -----------------------------------------------------------------

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_function, n_trials=15)

    # Obtener y guardar el mejor modelo
    best_model = get_best_model(experiment_id)  # elige el con mayor f1
    os.makedirs("models", exist_ok=True)  # crea carpeta models
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)  # Guarda modelo

    print(f"Mejor F1: {study.best_value:.4f}")

    return best_model


# Guardar archivo optimize
if __name__ == "__main__":
    optimize_model()
