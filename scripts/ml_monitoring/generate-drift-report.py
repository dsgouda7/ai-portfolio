#!/usr/bin/env python3
"""Generate ad-hoc drift analysis report using Evidently AI"""

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def generate_drift_report(reference_data_path, production_data_path, output_path):
    """Generate drift report comparing reference and production data"""
    # Load data
    reference = pd.read_csv(reference_data_path)
    production = pd.read_csv(production_data_path)
    
    # Generate report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    report.run(reference_data=reference, current_data=production)
    report.save_html(output_path)
    
    print(f"✅ Drift report saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python generate-drift-report.py <reference.csv> <production.csv> <output.html>")
        sys.exit(1)
    
    generate_drift_report(sys.argv[1], sys.argv[2], sys.argv[3])
