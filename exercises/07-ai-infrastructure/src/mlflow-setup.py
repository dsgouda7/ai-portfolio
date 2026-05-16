"""
MLflow experiment tracking and model registry setup
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Optional, Dict, Any
from .utils import setup_logging, load_config

logger = setup_logging()


class MLflowManager:
    """
    Manage MLflow tracking, experiments, and model registry
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize MLflow manager with configuration"""
        self.config = load_config(config_path)
        self.mlflow_config = self.config["mlflow"]
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        self.client = MlflowClient()
        
        logger.info(f"MLflow tracking URI: {self.mlflow_config['tracking_uri']}")
    
    def create_experiment(self, experiment_name: Optional[str] = None) -> str:
        """
        Create or get existing experiment
        
        Args:
            experiment_name: Name of experiment (uses config default if None)
        
        Returns:
            Experiment ID
        """
        name = experiment_name or self.mlflow_config["experiment_name"]
        
        try:
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=self.mlflow_config.get("default_artifact_root")
            )
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        except Exception:
            experiment = self.client.get_experiment_by_name(name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {name} (ID: {experiment_id})")
        
        return experiment_id
    
    def log_model_run(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        model_name: str = "model"
    ) -> str:
        """
        Log a model training run to MLflow
        
        Args:
            model: Trained model object
            params: Model hyperparameters
            metrics: Training metrics
            artifacts: Additional artifacts to log
            model_name: Name for the model
        
        Returns:
            Run ID
        """
        experiment_id = self.create_experiment()
        
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, artifact_path=model_name)
            
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)
            
            run_id = run.info.run_id
            logger.info(f"Logged run: {run_id}")
        
        return run_id
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a model in the MLflow Model Registry
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            model_name: Name for registered model
            tags: Optional tags for the model version
        
        Returns:
            Model version
        """
        try:
            # Register the model
            model_version = mlflow.register_model(model_uri, model_name)
            version = model_version.version
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=value
                    )
            
            logger.info(f"Registered model: {model_name} version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def promote_model_to_production(
        self,
        model_name: str,
        version: str
    ) -> None:
        """
        Promote a model version to production stage
        
        Args:
            model_name: Name of the registered model
            version: Model version to promote
        """
        try:
            # Archive current production models
            for mv in self.client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Archived"
                    )
            
            # Promote new version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            logger.info(f"Promoted {model_name} v{version} to Production")
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def load_production_model(self, model_name: str):
        """
        Load the current production model
        
        Args:
            model_name: Name of the registered model
        
        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded production model: {model_name}")
        return model
