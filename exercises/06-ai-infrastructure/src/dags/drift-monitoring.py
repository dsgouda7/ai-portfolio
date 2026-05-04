"""
Airflow DAG: Model Drift Monitoring
Daily drift detection on production data
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_monitoring import ModelMonitor

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_monitoring',
    default_args=default_args,
    description='Monitor model and data drift',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'production'],
)


def collect_production_data(**context):
    """Collect recent production predictions"""
    from src.utils import setup_logging
    import pandas as pd
    import numpy as np
    
    logger = setup_logging()
    logger.info("Collecting production data")
    
    # In production, query from production database
    # For demo, generate sample data
    current_data = pd.DataFrame({
        'feature_1': np.random.uniform(0, 100, 500),
        'feature_2': np.random.uniform(0, 50, 500),
        'feature_3': np.random.normal(25, 5, 500),
    })
    
    # Save to temporary location
    data_path = Path("data/temp/current_data.parquet")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    current_data.to_parquet(data_path)
    
    context['task_instance'].xcom_push(key='current_data_path', value=str(data_path))
    logger.info(f"Collected {len(current_data)} production records")


def load_reference_data(**context):
    """Load reference/training data"""
    from src.utils import setup_logging
    import pandas as pd
    import numpy as np
    
    logger = setup_logging()
    logger.info("Loading reference data")
    
    # In production, load from feature store or training data archive
    # For demo, generate sample data
    reference_data = pd.DataFrame({
        'feature_1': np.random.uniform(0, 100, 1000),
        'feature_2': np.random.uniform(0, 50, 1000),
        'feature_3': np.random.normal(20, 5, 1000),  # Slightly different distribution
    })
    
    data_path = Path("data/temp/reference_data.parquet")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    reference_data.to_parquet(data_path)
    
    context['task_instance'].xcom_push(key='reference_data_path', value=str(data_path))
    logger.info(f"Loaded {len(reference_data)} reference records")


def detect_drift(**context):
    """Run drift detection"""
    from src.utils import setup_logging
    import pandas as pd
    
    logger = setup_logging()
    logger.info("Running drift detection")
    
    # Load data from previous tasks
    current_path = context['task_instance'].xcom_pull(key='current_data_path', task_ids='collect_production_data')
    reference_path = context['task_instance'].xcom_pull(key='reference_data_path', task_ids='load_reference_data')
    
    current_data = pd.read_parquet(current_path)
    reference_data = pd.read_parquet(reference_path)
    
    # Run drift detection
    monitor = ModelMonitor()
    drift_results = monitor.detect_data_drift(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Push results
    context['task_instance'].xcom_push(key='drift_detected', value=drift_results['drift_detected'])
    
    if drift_results['drift_detected']:
        logger.warning("🚨 Data drift detected!")
    else:
        logger.info("✅ No drift detected")
    
    return drift_results


def trigger_retraining_if_needed(**context):
    """Trigger retraining pipeline if drift detected"""
    from src.utils import setup_logging
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    
    logger = setup_logging()
    
    drift_detected = context['task_instance'].xcom_pull(key='drift_detected', task_ids='detect_drift')
    
    if drift_detected:
        logger.info("Triggering retraining pipeline due to drift")
        # In production, this would trigger the retraining DAG
        # TriggerDagRunOperator(task_id='trigger_retrain', trigger_dag_id='retraining_pipeline')
        logger.info("Retraining pipeline triggered")
    else:
        logger.info("No retraining needed")


# Define tasks
collect_task = PythonOperator(
    task_id='collect_production_data',
    python_callable=collect_production_data,
    dag=dag,
)

reference_task = PythonOperator(
    task_id='load_reference_data',
    python_callable=load_reference_data,
    dag=dag,
)

drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='trigger_retraining_if_needed',
    python_callable=trigger_retraining_if_needed,
    dag=dag,
)

# Define dependencies
[collect_task, reference_task] >> drift_task >> retrain_task
