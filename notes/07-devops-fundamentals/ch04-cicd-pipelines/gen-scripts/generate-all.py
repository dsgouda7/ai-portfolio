"""
Generate all Ch.4 CI/CD Pipelines diagrams

Runs all diagram generation scripts in sequence.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a diagram generation script"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

if __name__ == '__main__':
    scripts = [
        'gen_ch04_cicd_pipeline.py',
        'gen_ch04_workflow_triggers.py',
        'gen_ch04_secrets_management.py'
    ]
    
    print("🎨 Generating Ch.4 CI/CD Pipelines diagrams...")
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {success_count}/{len(scripts)} diagram sets")
    print('='*60)
    
    if success_count == len(scripts):
        print("\nAll diagrams generated successfully!")
        print("Output location: ../img/")
    else:
        print("\n⚠️  Some diagrams failed to generate. Check errors above.")
        sys.exit(1)
