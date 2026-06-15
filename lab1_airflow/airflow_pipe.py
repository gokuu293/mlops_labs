import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train


def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/sonarsushant/California-House-Price-Prediction/master/housing.csv', delimiter=',')
    df.to_csv("housing.csv", index=False)
    print("df: ", df.shape)
    return True


def clear_data():
    df = pd.read_csv("housing.csv")

    cat_columns = ['ocean_proximity']
    num_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income',
                    'median_house_value']

    # заполнение пропусков медианой (total_bedrooms содержит NaN)
    for col in num_columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # здравый смысл: убираем строки с нулевыми/некорректными значениями
    df = df[df["median_house_value"] > 0]
    df = df[df["total_rooms"] > 0]
    df = df[df["households"] > 0]

    # анализ гистограмм: убираем выброс на верхней границе таргета (censored value 500001)
    df = df[df["median_house_value"] < 500001]

    df = df.reset_index(drop=True)

    # кодирование категориального признака
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]

    df.to_csv('df_clear.csv', index=False)
    print("df_clear: ", df.shape)
    return True


dag_house = DAG(
    dag_id="train_pipe_house",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(python_callable=download_data, task_id="download_house", dag=dag_house)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_house", dag=dag_house)
train_task = PythonOperator(python_callable=train, task_id="train_house", dag=dag_house)

download_task >> clear_task >> train_task
