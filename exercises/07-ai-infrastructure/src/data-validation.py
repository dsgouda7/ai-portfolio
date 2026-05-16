"""
Great Expectations data validation
"""
import great_expectations as gx
from great_expectations.data_context import DataContext
from great_expectations.checkpoint import SimpleCheckpoint
from pathlib import Path
from typing import Dict, Any, List
from .utils import setup_logging, load_config
import pandas as pd

logger = setup_logging()


class DataValidator:
    """
    Manage data validation with Great Expectations
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data validator"""
        self.config = load_config(config_path)
        self.validation_config = self.config["data_validation"]
        
        # Initialize data context
        self.context = self._initialize_context()
        logger.info("Data validator initialized")
    
    def _initialize_context(self) -> DataContext:
        """
        Initialize Great Expectations data context
        
        Returns:
            Configured DataContext
        """
        context_root = Path(self.validation_config["expectations_dir"]).parent
        context_root.mkdir(parents=True, exist_ok=True)
        
        try:
            context = gx.get_context(context_root_dir=str(context_root))
        except Exception:
            # Create new context if doesn't exist
            context = gx.get_context(mode="file")
        
        return context
    
    def create_expectation_suite(
        self,
        suite_name: str,
        expectations: List[Dict[str, Any]]
    ) -> None:
        """
        Create an expectation suite with specified expectations
        
        Args:
            suite_name: Name for the expectation suite
            expectations: List of expectation configurations
        """
        try:
            # Create or update suite
            suite = self.context.add_expectation_suite(
                expectation_suite_name=suite_name
            )
            
            # Add expectations
            for expectation in expectations:
                expectation_type = expectation.pop("expectation_type")
                suite.add_expectation(
                    expectation_type=expectation_type,
                    kwargs=expectation
                )
            
            self.context.save_expectation_suite(suite)
            logger.info(f"Created expectation suite: {suite_name}")
            
        except Exception as e:
            logger.error(f"Failed to create expectation suite: {e}")
            raise
    
    def create_default_suite(self, suite_name: str = "production_suite") -> None:
        """
        Create default expectation suite for ML data
        
        Args:
            suite_name: Name for the suite
        """
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "min_value": 100,
                "max_value": 1000000
            },
            {
                "expectation_type": "expect_table_column_count_to_equal",
                "value": 8  # Example: 7 features + 1 target
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "target"
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "feature_1",
                "min_value": 0,
                "max_value": 100
            },
            {
                "expectation_type": "expect_column_mean_to_be_between",
                "column": "feature_1",
                "min_value": 10,
                "max_value": 90
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "feature_1"
            }
        ]
        
        self.create_expectation_suite(suite_name, expectations)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame against an expectation suite
        
        Args:
            df: DataFrame to validate
            suite_name: Name of expectation suite
            checkpoint_name: Optional checkpoint name
        
        Returns:
            Validation results dictionary
        """
        checkpoint_name = checkpoint_name or self.validation_config["checkpoint_name"]
        
        try:
            # Create batch from dataframe
            batch = self.context.sources.add_pandas(
                name="validation_source"
            ).read_dataframe(df)
            
            # Run validation
            checkpoint = SimpleCheckpoint(
                name=checkpoint_name,
                data_context=self.context,
                expectation_suite_name=suite_name
            )
            
            results = checkpoint.run(batch_request=batch)
            
            # Extract results
            validation_result = {
                "success": results.success,
                "statistics": results.run_results,
                "evaluated_expectations": len(results.run_results)
            }
            
            if results.success:
                logger.info(f"Validation passed: {suite_name}")
            else:
                logger.warning(f"Validation failed: {suite_name}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise
    
    def validate_file(
        self,
        file_path: str,
        suite_name: str,
        checkpoint_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a file against an expectation suite
        
        Args:
            file_path: Path to data file
            suite_name: Name of expectation suite
            checkpoint_name: Optional checkpoint name
        
        Returns:
            Validation results dictionary
        """
        # Read file and validate
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_parquet(file_path)
        return self.validate_dataframe(df, suite_name, checkpoint_name)
