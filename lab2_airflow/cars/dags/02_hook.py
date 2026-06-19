from datetime import datetime
import json
import logging
import os

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from hooks import CarsHook  # ← импортируем из plugins/hooks/

# from hooks import MovielensHook

# with DAG(
#     dag_id="02_hook",
#     description="Fetches ratings from the Movielens API using a custom hook.",
#     start_date=datetime(2023, 1, 1),
#     end_date=datetime(2023, 1, 10),
#     schedule=CronDataIntervalTimetable("@daily", "UTC"),
#     catchup=True,
# ):

#     def _fetch_ratings(conn_id:str, templates_dict:dict, batch_size:int=1000, **_):
#         logger = logging.getLogger(__name__)

#         start_date = templates_dict["start_date"]
#         end_date = templates_dict["end_date"]
#         output_path = templates_dict["output_path"]

#         logger.info(f"Fetching ratings for {start_date} to {end_date}")
#         hook = MovielensHook(conn_id=conn_id)
#         ratings = list(hook.get_ratings(start_date=start_date, end_date=end_date, batch_size=batch_size))
#         logger.info(f"Fetched {len(ratings)} ratings")

#         logger.info(f"Writing ratings to {output_path}")

#         # Make sure output directory exists.
#         output_dir = os.path.dirname(output_path)
#         os.makedirs(output_dir, exist_ok=True)

#         with open(output_path, "w") as file_:
#             json.dump(ratings, fp=file_)

#     PythonOperator(
#         task_id="fetch_ratings",
#         python_callable=_fetch_ratings,
#         op_kwargs={"conn_id": "movielens"},
#         templates_dict={
#             "start_date": "{{data_interval_start | ds}}",
#             "end_date": "{{data_interval_end | ds}}",
#             "output_path": "/data/custom_hook/{{data_interval_start | ds}}.json",
#         },
#     )


def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    # Убедимся, что директория существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved cars to {output_path}")


def clean_cars_data(**kwargs):
    raw_file_path = '/data/custom_hook/cars.json'
    cleaned_file_path = '/data/cleaned/cars_cleaned.json'

    print(f"Loading raw data from {raw_file_path}")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(
            f"File {raw_file_path} not found")

    with open(raw_file_path, "r") as file_:
        cars = json.load(file_)

    print(f"Initial data size: {len(cars)}")

    unique_cars = []
    seen = set()
    for car in cars:
        key = json.dumps(car, sort_keys=True)
        if key not in seen:
            unique_cars.append(car)
            seen.add(key)

    print(f"Data size after dropping duplicates: {len(unique_cars)}")

    cleaned_cars = [
        car for car in unique_cars
        if all(value is not None and value != "" for value in car.values())
    ]

    print(f"Data size after dropping nulls: {len(cleaned_cars)}")

    categorical_columns = ("Fuel_type", "Transmission")
    encoders = {column: {} for column in categorical_columns}

    for car in cleaned_cars:
        for column in categorical_columns:
            value = car.get(column)
            if value not in encoders[column]:
                encoders[column][value] = len(encoders[column])
            car[column] = encoders[column][value]

    print(f"Categorical features transformed: {categorical_columns}")

    os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)

    with open(cleaned_file_path, "w") as file_:
        json.dump(cleaned_cars, file_, indent=4)

    print(f"Cleaned data saved to {cleaned_file_path}")


with DAG(
    dag_id="02_hook",
    description="Fetches car data from the custom API using a custom hook.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_cars_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},  # ← имя Airflow Connection
        templates_dict={
            "output_path": "/data/custom_hook/cars.json",
        },
    )
    clean_cars_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=clean_cars_data,
    )

    fetch_cars_task >> clean_cars_task
