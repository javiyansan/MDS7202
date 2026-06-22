import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path(
    "/Users/javierayanez/Documents/Semestre 11/MDS7202-Laboratorio_Programacion_Cientifica/Studio J - MDS7202/labs/lab_9/data"
)  # AJUSTA esta ruta
OUTPUT_PATH = Path("/tmp/spotify_data.parquet")

PARAM_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "tempo",
    "duration_ms",
    "year",
]


# ── Funciones auxiliares (dadas) ─────────────────────────────────────────────


def load_batch(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_all_parallel(data_dir: Path, n_batches: int = 5) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.parquet"))[:n_batches]
    with ThreadPoolExecutor(max_workers=None) as executor:
        dfs = list(executor.map(load_batch, [str(p) for p in paths]))
    return pd.concat(dfs, ignore_index=True)


def build_pipeline(n_jobs: int = -1) -> Pipeline:
    return Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        ("ohe", OneHotEncoder(handle_unknown="ignore"), ["key", "mode", "genre"]),
                        ("numerical", "passthrough", PARAM_COLS),
                    ]
                ),
            ),
            ("random_forest", RandomForestRegressor(n_jobs=n_jobs, random_state=42)),
        ]
    )


# ── Funciones de las tareas de Airflow ───────────────────────────────────────


def task_load_data_fn(**context):
    """
    Carga 5 batches de datos en paralelo y guarda el resultado en disco.
    TODO: implementa esta función.
    - Usa load_all_parallel para cargar los datos.
    - Guarda el DataFrame resultante en OUTPUT_PATH (formato parquet).
    - Usa XCom para pasar la ruta del archivo a la siguiente tarea.
    """

    df = load_all_parallel(DATA_DIR, n_batches=5)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # guardado en formato parquet
    df.to_parquet(OUTPUT_PATH, index=False)

    # Pasar la ruta a la siguiente tarea via XCom
    context["ti"].xcom_push(key="data_path", value=str(OUTPUT_PATH))


def task_train_model_fn(**context):
    """
    Carga los datos desde disco y entrena el pipeline.
    TODO: implementa esta función.
    - Recupera la ruta del archivo desde XCom.
    - Lee el DataFrame desde esa ruta.
    - Prepara X e y, realiza el split 80/20.
    - Entrena build_pipeline(n_jobs=-1).
    - Imprime el tiempo de entrenamiento.
    """
    # Recuperar ruta del archivo
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="load_data")

    df = pd.read_parquet(data_path)  # leer df

    X = df[PARAM_COLS + ["key", "mode", "genre"]]
    y = df["valence"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split

    # Entrenamiento
    pipeline = build_pipeline(n_jobs=-1)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"Tiempo de entrenamiento: {elapsed:.2f}s")


# ── Definición del DAG ────────────────────────────────────────────────────────

with DAG(
    dag_id="spotify_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mds7202", "spotify"],
) as dag:
    load_data = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data_fn,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model_fn,
    )

    # TODO: define la dependencia entre tareas (load_data debe ejecutarse antes que train_model)
    load_data >> train_model
