"""
Airflow DAG: ML Training Pipeline
Orchestrates data → train → validate → register workflow
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlflow_setup import MLflowManager
from src.data_validation import DataValidator
from src.feature_store import FeatureStoreManager

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'production'],
)


def extract_features(**context):
    """Extract features from feature store"""
    from src.utils import setup_logging
    import pandas as pd
    
    logger = setup_logging()
    logger.info("Extracting features from feature store")
    
    # In production, this would query the feature store
    # For demo, generate sample data
    feature_store = FeatureStoreManager()
    
    # Push feature data to XCom for next task
    context['task_instance'].xcom_push(key='features_extracted', value=True)
    logger.info("Features extracted successfully")


def validate_data(**context):
    """Validate input data quality"""
    from src.utils import setup_logging
    import pandas as pd
    import numpy as np
    
    logger = setup_logging()
    logger.info("Validating data quality")
    
    # Create sample data for validation
    data = pd.DataFrame({
        'target': np.random.randn(1000),
        'feature_1': np.random.uniform(0, 100, 1000),
    })
    
    validator = DataValidator()
    
    # Create and run validation suite
    validator.create_default_suite()
    results = validator.validate_dataframe(data, "production_suite")
    
    if not results['success']:
        raise ValueError("Data validation failed!")
    
    logger.info("Data validation passed")
    context['task_instance'].xcom_push(key='data_validated', value=True)


def train_model(**context):
    """Train ML model"""
    from src.utils import setup_logging
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import joblib
    
    logger = setup_logging()
    logger.info("Training model")
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    # Save model
    model_path = "models/production_model_latest.pkl"
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    
    context['task_instance'].xcom_push(key='model_path', value=model_path)
    context['task_instance'].xcom_push(key='test_score', value=test_score)
    logger.info(f"Model saved: {model_path}")


def register_model(**context):
    """Register model in MLflow registry"""
    from src.utils import setup_logging
    import joblib
    
    logger = setup_logging()
    logger.info("Registering model in MLflow")
    
    # Get model path from previous task
    model_path = context['task_instance'].xcom_pull(key='model_path', task_ids='train_model')
    test_score = context['task_instance'].xcom_pull(key='test_score', task_ids='train_model')
    
    # Load model
    model = joblib.load(model_path)
    
    # Log to MLflow
    mlflow_manager = MLflowManager()
    
    params = {
        'n_estimators': 100,
        'random_state': 42
    }
    
    metrics = {
        'test_r2_score': test_score
    }
    
    run_id = mlflow_manager.log_model_run(
        model=model,
        params=params,
        metrics=metrics,
        model_name="production_model"
    )
    
    # Register model
    model_uri = f"runs:/{run_id}/production_model"
    version = mlflow_manager.register_model(
        model_uri=model_uri,
        model_name="production_model",
        tags={"stage": "candidate", "pipeline": "airflow"}
    )
    
    logger.info(f"Model registered: version {version}")


def promote_to_production(**context):
    """Promote model to production if metrics meet threshold"""
    from src.utils import setup_logging
    
    logger = setup_logging()
    logger.info("Evaluating model for production promotion")
    
    test_score = context['task_instance'].xcom_pull(key='test_score', task_ids='train_model')
    
    # Production threshold
    PROD_THRESHOLD = 0.85
    
    if test_score >= PROD_THRESHOLD:
        mlflow_manager = MLflowManager()
        # Get latest version
        versions = mlflow_manager.client.search_model_versions(
            filter_string="name='production_model'"
        )
        latest_version = max([int(v.version) for v in versions])
        
        mlflow_manager.promote_model_to_production(
            model_name="production_model",
            version=str(latest_version)
        )
        logger.info(f"✅ Model promoted to production (R²={test_score:.4f})")
    else:
        logger.warning(f"⚠️ Model not promoted - score {test_score:.4f} < {PROD_THRESHOLD}")


# Define tasks
extract_task = PythonOperator(
    task_id='extract_features',
    python_callable=extract_features,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

promote_task = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag,
)

# Define dependencies
extract_task >> validate_task >> train_task >> register_task >> promote_task
