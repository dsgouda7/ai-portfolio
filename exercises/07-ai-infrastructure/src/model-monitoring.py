"""
Model monitoring with Evidently AI for drift detection
"""
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from pathlib import Path
from typing import Optional, Dict, Any
from .utils import setup_logging, load_config
import pandas as pd
import json

logger = setup_logging()


class ModelMonitor:
    """
    Monitor model performance and data drift
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model monitor"""
        self.config = load_config(config_path)
        self.monitoring_config = self.config["model_monitoring"]
        
        # Create monitoring directory
        self.metrics_store = Path(self.monitoring_config["metrics_store"])
        self.metrics_store.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model monitor initialized")
    
    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data
        
        Args:
            reference_data: Baseline/training data
            current_data: Production/current data
            column_mapping: Optional column mapping configuration
        
        Returns:
            Drift detection results
        """
        try:
            # Create drift report
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Extract drift metrics
            report_dict = report.as_dict()
            
            # Check drift threshold
            drift_score = report_dict.get("metrics", [{}])[0].get("result", {}).get("dataset_drift", False)
            drift_detected = drift_score if isinstance(drift_score, bool) else False
            
            results = {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "timestamp": pd.Timestamp.now().isoformat(),
                "reference_size": len(reference_data),
                "current_size": len(current_data)
            }
            
            # Save report
            report_path = self.metrics_store / f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            if drift_detected:
                logger.warning(f"Data drift detected! Report saved: {report_path}")
                if self.monitoring_config.get("alert_on_drift"):
                    self._send_drift_alert(results)
            else:
                logger.info("No data drift detected")
            
            return results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise
    
    def detect_target_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str,
        column_mapping: Optional[ColumnMapping] = None
    ) -> Dict[str, Any]:
        """
        Detect target/prediction drift
        
        Args:
            reference_data: Baseline data with target
            current_data: Current data with target
            target_column: Name of target column
            column_mapping: Optional column mapping
        
        Returns:
            Target drift results
        """
        try:
            # Ensure column mapping includes target
            if column_mapping is None:
                column_mapping = ColumnMapping(target=target_column)
            
            # Create target drift report
            report = Report(metrics=[
                TargetDriftPreset(),
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            report_dict = report.as_dict()
            
            results = {
                "target_drift_detected": report_dict.get("metrics", [{}])[0].get("result", {}).get("drift_detected", False),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Save report
            report_path = self.metrics_store / f"target_drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            logger.info(f"Target drift report saved: {report_path}")
            return results
            
        except Exception as e:
            logger.error(f"Target drift detection failed: {e}")
            raise
    
    def run_drift_test_suite(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        column_mapping: Optional[ColumnMapping] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive drift test suite
        
        Args:
            reference_data: Baseline data
            current_data: Current data
            column_mapping: Optional column mapping
        
        Returns:
            Test results
        """
        try:
            # Create test suite
            test_suite = TestSuite(tests=[
                DataDriftTestPreset(),
            ])
            
            test_suite.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Get results
            test_results = test_suite.as_dict()
            
            results = {
                "all_tests_passed": test_results.get("summary", {}).get("all_passed", False),
                "total_tests": test_results.get("summary", {}).get("total_tests", 0),
                "success_tests": test_results.get("summary", {}).get("success_tests", 0),
                "failed_tests": test_results.get("summary", {}).get("failed_tests", 0)
            }
            
            # Save test report
            report_path = self.metrics_store / f"drift_tests_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            test_suite.save_html(str(report_path))
            
            logger.info(f"Drift test suite completed: {results['success_tests']}/{results['total_tests']} passed")
            return results
            
        except Exception as e:
            logger.error(f"Drift test suite failed: {e}")
            raise
    
    def _send_drift_alert(self, drift_results: Dict[str, Any]) -> None:
        """
        Send alert when drift is detected
        
        Args:
            drift_results: Drift detection results
        """
        # In production, this would integrate with alerting systems
        # (e.g., PagerDuty, Slack, email)
        alert_message = f"""
        🚨 DATA DRIFT ALERT 🚨
        
        Drift detected in production data!
        - Timestamp: {drift_results['timestamp']}
        - Drift Score: {drift_results.get('drift_score', 'N/A')}
        - Current data size: {drift_results['current_size']}
        
        Action required: Review model performance and consider retraining.
        """
        
        logger.warning(alert_message)
        
        # Save alert to file
        alert_path = self.metrics_store / "alerts.json"
        alerts = []
        if alert_path.exists():
            with open(alert_path, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(drift_results)
        
        with open(alert_path, 'w') as f:
            json.dump(alerts, f, indent=2)
