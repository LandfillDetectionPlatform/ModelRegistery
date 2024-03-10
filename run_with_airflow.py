from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from pipeline.training_pipeline import image_classification_pipeline
from pipeline.inference_pipeline import inference_pipeline

def run_training_pipeline():
    pipeline = image_classification_pipeline()
    pipeline.run()

def run_inference_pipeline():
    pipeline = inference_pipeline()
    pipeline.run()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 4),
    'retries': 1,
}

dag = DAG('image_classification_workflow',
          default_args=default_args,
          description='A simple image classification workflow',
          schedule_interval='@daily',
          )

training_operator = PythonOperator(task_id='training_task',
                                   python_callable=run_training_pipeline,
                                   dag=dag)

inference_operator = PythonOperator(task_id='inference_task',
                                    python_callable=run_inference_pipeline,
                                    dag=dag)

training_operator >> inference_operator
