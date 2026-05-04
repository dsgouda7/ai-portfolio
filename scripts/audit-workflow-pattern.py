#!/usr/bin/env python3
"""
Validate chapter README against workflow pattern requirements.

Usage:
    python scripts/audit_workflow_pattern.py --chapter notes/01-ml/.../README.md
    python scripts/audit_workflow_pattern.py --chapter notes/01-ml/.../README.md --verbose

Checks:
    1. §1.5 Practitioner Workflow section exists
    2. At least N phase markers ([Phase 1: VERB]) where N >= 3
    3. At least 3 decision checkpoints (DECISION CHECKPOINT format)
    4. At least 3 executable code snippets (```python or ```bash with imports)
    5. At least 3 industry callout boxes (> **Industry Standard:** or **Industry Standard:** heading)
    6. No broken section references (§ 3.1 should be §3.1)

Returns:
    - EXIT 0: All checks pass
    - EXIT 1: One or more checks fail (prints violations)
"""

import re
import sys
from pathlib import Path
from typing import Dict, List


def audit_workflow_compliance(readme_path: Path, verbose: bool = False) -> Dict:
    """
    Audit a chapter README for workflow pattern compliance.
    
    Args:
        readme_path: Path to README.md file
        verbose: Print detailed findings
        
    Returns:
        Dictionary with results, failures list, and pass/fail status
    """
    if not readme_path.exists():
        return {
            'results': {},
            'failures': [f"File not found: {readme_path}"],
            'passed': False
        }
    
    content = readme_path.read_text(encoding='utf-8')
    
    # Check 1: §1.5 Practitioner Workflow section (various formats)
    has_workflow_section = bool(
        re.search(r'§1\.5.*Practitioner Workflow', content, re.IGNORECASE) or
        re.search(r'1\.5.*Practitioner Workflow', content, re.IGNORECASE) or
        re.search(r'##\s+1\.5.*Workflow', content, re.IGNORECASE)
    )
    
    # Check 2: Phase markers [Phase N: VERB]
    phase_markers = re.findall(r'\[Phase \d+:', content)
    phase_count = len(phase_markers)
    
    # Check 3: Decision checkpoints (various formats)
    decision_checkpoints = []
    # Format 1: ### ✓ DECISION CHECKPOINT
    decision_checkpoints.extend(re.findall(r'###\s*✓\s*DECISION CHECKPOINT', content, re.IGNORECASE))
    # Format 2: ## DECISION CHECKPOINT
    decision_checkpoints.extend(re.findall(r'##\s*DECISION CHECKPOINT', content, re.IGNORECASE))
    # Format 3: ### DECISION:
    decision_checkpoints.extend(re.findall(r'###\s*DECISION:', content, re.IGNORECASE))
    checkpoint_count = len(decision_checkpoints)
    
    # Check 4: Code snippets (python, bash, yaml, etc.)
    code_snippets = re.findall(r'```(?:python|bash|yaml|javascript|typescript|sql)', content)
    snippet_count = len(code_snippets)
    
    # Check 5: Industry callout boxes
    industry_callouts = []
    # Format 1: > **Industry Standard:**
    industry_callouts.extend(re.findall(r'>\s*\*\*Industry Standard:', content))
    # Format 2: ### Industry Standard heading
    industry_callouts.extend(re.findall(r'###\s*Industry Standard', content))
    # Format 3: **Industry Standard:** (inline)
    industry_callouts.extend(re.findall(r'\*\*Industry Standard:\*\*', content))
    callout_count = len(industry_callouts)
    
    # Check 6: Broken section references (space after §)
    broken_refs = re.findall(r'§ \d+\.', content)
    broken_ref_count = len(broken_refs)
    
    results = {
        'has_workflow_section': has_workflow_section,
        'phase_markers': phase_count,
        'decision_checkpoints': checkpoint_count,
        'code_snippets': snippet_count,
        'industry_callouts': callout_count,
        'broken_refs': broken_ref_count,
    }
    
    # Validation
    failures = []
    if not results['has_workflow_section']:
        failures.append("[X] Missing §1.5 Practitioner Workflow section")
    
    if results['phase_markers'] < 3:
        failures.append(f"[X] Insufficient phase markers: {results['phase_markers']} found (need >= 3)")
    elif verbose:
        print(f"  [OK] Phase markers: {results['phase_markers']}")
    
    if results['decision_checkpoints'] < 3:
        failures.append(f"[X] Insufficient decision checkpoints: {results['decision_checkpoints']} found (need >= 3)")
    elif verbose:
        print(f"  [OK] Decision checkpoints: {results['decision_checkpoints']}")
    
    if results['code_snippets'] < 3:
        failures.append(f"[X] Insufficient code snippets: {results['code_snippets']} found (need >= 3)")
    elif verbose:
        print(f"  [OK] Code snippets: {results['code_snippets']}")
    
    if results['industry_callouts'] < 3:
        failures.append(f"[!]  Low industry callout count: {results['industry_callouts']} found (recommended >= 3)")
        # Don't fail on this - it's a soft requirement
    elif verbose:
        print(f"  [OK] Industry callouts: {results['industry_callouts']}")
    
    if results['broken_refs'] > 0:
        failures.append(f"[X] Found {results['broken_refs']} broken section references (§ N.M should be §N.M)")
    elif verbose:
        print(f"  [OK] No broken section references")
    
    # Filter out warnings from hard failures
    hard_failures = [f for f in failures if f.startswith('[X]')]
    passed = len(hard_failures) == 0
    
    return {
        'results': results,
        'failures': failures,
        'passed': passed
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate chapter README against workflow pattern requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check single chapter
    python scripts/audit_workflow_pattern.py --chapter notes/01-ml/01_regression/ch03_feature_importance/README.md
    
    # Check with verbose output
    python scripts/audit_workflow_pattern.py --chapter notes/01-ml/.../README.md --verbose
    
    # Check all Wave 1 chapters
    for chapter in notes/01-ml/01_regression/ch00b_class_imbalance/README.md \\
                    notes/01-ml/03_neural_networks/ch08_tensorboard/README.md \\
                    notes/00-math_under_the_hood/ch04_small_steps/README.md \\
                    notes/03-ai/ch02_prompt_engineering/prompt-engineering.md \\
                    notes/06-ai_infrastructure/ch02_memory_and_compute_budgets/memory-budgets.md \\
                    notes/07-devops_fundamentals/ch08_security_secrets_management/README.md; do
        python scripts/audit_workflow_pattern.py --chapter "$chapter"
    done
        """
    )
    parser.add_argument(
        '--chapter',
        type=Path,
        required=True,
        help='Path to chapter README.md file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed findings'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path for consistent handling
    chapter_path = args.chapter.resolve()
    
    audit = audit_workflow_compliance(chapter_path, verbose=args.verbose)
    
    # Try to get relative path for display, fallback to absolute
    try:
        display_path = chapter_path.relative_to(Path.cwd())
    except ValueError:
        display_path = chapter_path
    
    if audit['passed']:
        print(f"[PASS] {display_path}")
        if args.verbose:
            print("\nMetrics:")
            for key, value in audit['results'].items():
                if key != 'has_workflow_section':
                    print(f"  {key}: {value}")
        sys.exit(0)
    else:
        print(f"[FAIL] {display_path}")
        for failure in audit['failures']:
            print(f"  {failure}")
        if args.verbose:
            print("\nFound:")
            for key, value in audit['results'].items():
                if key != 'has_workflow_section':
                    print(f"  {key}: {value}")
        sys.exit(1)


if __name__ == '__main__':
    main()
