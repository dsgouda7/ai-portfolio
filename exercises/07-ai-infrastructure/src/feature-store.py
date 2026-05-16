"""
Feast Feature Store configuration and management
"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.feature_store import FeatureStore
from pathlib import Path
from typing import List, Dict, Any
from .utils import setup_logging, load_config
import pandas as pd

logger = setup_logging()


class FeatureStoreManager:
    """
    Manage feature definitions and materialization
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature store manager"""
        self.config = load_config(config_path)
        self.fs_config = self.config["feature_store"]
        
        # Initialize feature store
        self.store = FeatureStore(repo_path=".")
        logger.info("Feature store initialized")
    
    def create_feature_definitions(self) -> None:
        """
        Define feature entities and views
        Creates example feature definitions for ML model
        """
        # Define entity (e.g., user, property, etc.)
        property_entity = Entity(
            name="property_id",
            value_type=ValueType.INT64,
            description="Property identifier"
        )
        
        # Define feature source
        property_features_source = FileSource(
            path=str(Path(self.fs_config["offline_store_path"]) / "property_features.parquet"),
            timestamp_field="event_timestamp",
        )
        
        # Define feature view
        property_features_view = FeatureView(
            name="property_features",
            entities=["property_id"],
            ttl=timedelta(days=30),
            features=[
                Feature(name="square_feet", dtype=ValueType.FLOAT),
                Feature(name="num_bedrooms", dtype=ValueType.INT64),
                Feature(name="num_bathrooms", dtype=ValueType.FLOAT),
                Feature(name="year_built", dtype=ValueType.INT64),
                Feature(name="location_score", dtype=ValueType.FLOAT),
                Feature(name="school_rating", dtype=ValueType.FLOAT),
                Feature(name="crime_rate", dtype=ValueType.FLOAT),
            ],
            online=True,
            source=property_features_source,
            tags={"team": "ml-engineering"}
        )
        
        logger.info("Feature definitions created")
    
    def materialize_features(
        self,
        start_date: str,
        end_date: str,
        feature_views: Optional[List[str]] = None
    ) -> None:
        """
        Materialize features to online store
        
        Args:
            start_date: Start date for materialization (ISO format)
            end_date: End date for materialization (ISO format)
            feature_views: List of feature view names (None = all)
        """
        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views
            )
            logger.info(f"Materialized features from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Feature materialization failed: {e}")
            raise
    
    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        feature_refs: List[str]
    ) -> pd.DataFrame:
        """
        Retrieve features from online store
        
        Args:
            entity_rows: List of entity dictionaries (e.g., [{"property_id": 123}])
            feature_refs: List of feature references (e.g., ["property_features:square_feet"])
        
        Returns:
            DataFrame with requested features
        """
        try:
            features_dict = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            ).to_dict()
            
            df = pd.DataFrame(features_dict)
            logger.info(f"Retrieved {len(df)} feature records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve online features: {e}")
            raise
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str]
    ) -> pd.DataFrame:
        """
        Retrieve historical features for training
        
        Args:
            entity_df: DataFrame with entity_id and event_timestamp
            feature_refs: List of feature references
        
        Returns:
            DataFrame with historical features
        """
        try:
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            
            logger.info(f"Retrieved {len(training_df)} historical feature records")
            return training_df
            
        except Exception as e:
            logger.error(f"Failed to retrieve historical features: {e}")
            raise
