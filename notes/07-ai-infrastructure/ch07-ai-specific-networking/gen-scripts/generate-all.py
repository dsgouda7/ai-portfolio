#!/usr/bin/env python3
"""
Master script to generate all diagrams for AI-Specific Networking chapter.
Runs all generation scripts and creates the img/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run all generation scripts."""
    # Get paths
    script_dir = Path(__file__).parent
    img_dir = script_dir.parent / 'img'
    
    # Create img directory
    img_dir.mkdir(exist_ok=True)
    print(f"✅ Created directory: {img_dir}\n")
    
    # List of generation scripts
    scripts = [
        'gen_network_topology.py',
        'gen_bandwidth_comparison.py',
        'gen_latency_heatmap.py',
        'gen_decision_tree.py'
    ]
    
    print("="*80)
    print("Generating AI-Specific Networking Diagrams")
    print("="*80)
    print()
    
    # Run each script
    for script in scripts:
        script_path = script_dir / script
        
        if not script_path.exists():
            print(f"⚠️  Script not found: {script}")
            continue
        
        print(f"Running {script}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error running {script}:")
                print(result.stderr)
        
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout running {script}")
        except Exception as e:
            print(f"❌ Exception running {script}: {e}")
        
        print()
    
    print("="*80)
    print("✅ All diagrams generated!")
    print(f"   Output directory: {img_dir}")
    print("="*80)
    
    # List generated files
    generated_files = sorted(img_dir.glob('*.png'))
    if generated_files:
        print("\nGenerated files:")
        for f in generated_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("\n⚠️  No PNG files found in output directory")

if __name__ == '__main__':
    main()
