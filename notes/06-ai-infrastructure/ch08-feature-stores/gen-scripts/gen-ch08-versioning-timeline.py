"""
Generate feature versioning timeline diagram.

Shows how feature definitions evolve over time and how Git-based versioning
enables reproducibility of training data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from datetime import datetime, timedelta

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 10), facecolor='#1a1a2e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

colors = {
    'primary': '#1e3a8a',
    'success': '#15803d',
    'caution': '#b45309',
    'danger': '#b91c1c',
    'info': '#1d4ed8'
}

# Title
ax.text(5, 11.5, 'Feature Versioning Timeline', ha='center', va='top',
        fontsize=18, fontweight='bold', color='white')

# Draw timeline
timeline_y = 8
ax.plot([1, 9], [timeline_y, timeline_y], color='white', linewidth=3, alpha=0.8)

# Timeline events
events = [
    {
        'x': 1.5, 'date': 'Jan 1', 'color': colors['info'],
        'title': 'Feature v1',
        'details': [
            'user_avg_purchase',
            'Definition: AVG(amount)',
            'Includes ALL transactions',
            'Git tag: features-v1'
        ]
    },
    {
        'x': 3.5, 'date': 'Jan 15', 'color': colors['success'],
        'title': 'Model v1 Trained',
        'details': [
            'Using features-v1',
            '94% accuracy',
            'MLflow run: 7a3f9b2',
            'Git commit logged'
        ]
    },
    {
        'x': 5.5, 'date': 'Feb 1', 'color': colors['caution'],
        'title': 'Feature v2',
        'details': [
            'user_avg_purchase',
            'Definition: AVG(amount)',
            'WHERE amount > 0',
            'Git tag: features-v2'
        ]
    },
    {
        'x': 7.5, 'date': 'Feb 15', 'color': colors['danger'],
        'title': 'Model v1 Retrain',
        'details': [
            'Checkout features-v1',
            'Reproduce exact data',
            'Same 94% accuracy ✅',
            'Validates reproducibility'
        ]
    }
]

for event in events:
    # Draw marker
    circle = Circle((event['x'], timeline_y), 0.15, color=event['color'], zorder=10)
    ax.add_patch(circle)
    
    # Draw vertical line
    ax.plot([event['x'], event['x']], [timeline_y, timeline_y + 1.2], 
            color=event['color'], linewidth=2, alpha=0.8)
    
    # Date label
    ax.text(event['x'], timeline_y - 0.4, event['date'],
           ha='center', fontsize=10, color='white', fontweight='bold')
    
    # Event box
    box_y = timeline_y + 1.3
    box = FancyBboxPatch(
        (event['x'] - 0.6, box_y), 1.2, 2.2,
        boxstyle="round,pad=0.1",
        facecolor=event['color'],
        edgecolor='white',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Title
    ax.text(event['x'], box_y + 2.0, event['title'],
           ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    # Details
    detail_y = box_y + 1.6
    for detail in event['details']:
        ax.text(event['x'], detail_y, detail,
               ha='center', va='top', fontsize=8, color='white')
        detail_y -= 0.35

# Add arrows showing relationships
# v1 → Model v1
ax.annotate('', xy=(3.5, 9.5), xytext=(1.5, 9.5),
           arrowprops=dict(arrowstyle='->', color='white', lw=2, alpha=0.6))
ax.text(2.5, 9.8, 'trained with', ha='center', fontsize=9, color='white',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='white'))

# v2 deployed (different definition)
ax.annotate('', xy=(5.5, 9.5), xytext=(3.5, 9.5),
           arrowprops=dict(arrowstyle='->', color=colors['caution'], lw=2, alpha=0.6))
ax.text(4.5, 9.8, '⚠️ definition changed', ha='center', fontsize=9, color=colors['caution'],
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=colors['caution']),
       fontweight='bold')

# Retrain with v1 (reproduce)
ax.annotate('', xy=(7.5, 9.5), xytext=(5.5, 9.5),
           arrowprops=dict(arrowstyle='->', color='white', lw=2, alpha=0.6))
ax.text(6.5, 9.8, 'git checkout v1', ha='center', fontsize=9, color='white',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor='white'))

# Problem scenario (no versioning)
problem_y = 5.5
ax.text(5, problem_y + 1.2, '❌ Without Feature Versioning', ha='center', fontsize=12,
       color=colors['danger'], fontweight='bold')

problem_box = FancyBboxPatch(
    (0.5, problem_y - 1.5), 9, 2.5,
    boxstyle="round,pad=0.2",
    facecolor=colors['danger'],
    edgecolor='white',
    linewidth=2,
    alpha=0.3
)
ax.add_patch(problem_box)

problem_steps = [
    '1. Train model on Jan 1 → 94% accuracy (using feature definition v1)',
    '2. Update feature definition on Feb 1 (v2: exclude $0 transactions)',
    '3. Production uses v2 → training-serving skew → accuracy drops to 89%',
    '4. Try to retrain model → can\'t reproduce original 94% (v1 definition lost)'
]

step_y = problem_y + 0.8
for step in problem_steps:
    ax.text(0.7, step_y, step, fontsize=10, color='white', va='top')
    step_y -= 0.5

# Solution scenario (with versioning)
solution_y = 2.5
ax.text(5, solution_y + 1.2, '✅ With Git-Based Feature Versioning', ha='center', fontsize=12,
       color=colors['success'], fontweight='bold')

solution_box = FancyBboxPatch(
    (0.5, solution_y - 1.5), 9, 2.5,
    boxstyle="round,pad=0.2",
    facecolor=colors['success'],
    edgecolor='white',
    linewidth=2,
    alpha=0.3
)
ax.add_patch(solution_box)

solution_steps = [
    '1. Train model on Jan 1 → git tag features-v1 → log commit SHA in MLflow',
    '2. Update feature definition on Feb 1 → git tag features-v2',
    '3. Production pins to features-v1 → no skew → 94% accuracy maintained',
    '4. Retrain model → git checkout features-v1 → reproduce exact 94% training data ✅'
]

step_y = solution_y + 0.8
for step in solution_steps:
    ax.text(0.7, step_y, step, fontsize=10, color='white', va='top')
    step_y -= 0.5

# Key insight box
insight_box = FancyBboxPatch(
    (0.5, 0.2), 9, 0.8,
    boxstyle="round,pad=0.2",
    facecolor=colors['info'],
    edgecolor='white',
    linewidth=3,
    alpha=0.9
)
ax.add_patch(insight_box)

ax.text(5, 0.7, '💡 Key Insight: Git-versioned features + MLflow tracking = 100% reproducibility',
       ha='center', fontsize=11, color='white', fontweight='bold')
ax.text(5, 0.4, 'Every model version has an immutable snapshot of feature definitions',
       ha='center', fontsize=10, color='white')

plt.tight_layout()
plt.savefig('../img/ch08-feature-versioning-timeline.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Generated: ch08-feature-versioning-timeline.png")
plt.close()
