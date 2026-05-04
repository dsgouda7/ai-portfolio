"""
Generate Feature Candidacy Decision Flow Animation for ML Ch.3
Creates a slow-motion animation showing 4 example features flowing through 
the diagnostic decision tree.

Features demonstrated:
1. MedInc → Strong independent predictor (M1 high, VIF low, M3 high)
2. Lat/Lon → Jointly irreplaceable (M1 low, M2/M3 high, Δ_interact > 0)
3. AveRooms → Collinear signal (VIF > 5)
4. Population → Drop candidate (all metrics near-zero)

Run from project root: python scripts/generate_feature_candidacy_flow.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Dark theme
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'text.color': '#e2e8f0',
    'xtick.color': '#e2e8f0',
    'ytick.color': '#e2e8f0',
})


class DecisionNode:
    """Represents a node in the decision tree"""
    def __init__(self, x, y, text, node_type='decision', color='#3b82f6'):
        self.x = x
        self.y = y
        self.text = text
        self.node_type = node_type  # 'start', 'decision', 'outcome'
        self.color = color
        self.width = 2.5 if node_type == 'decision' else 3.0
        self.height = 1.0 if node_type == 'decision' else 1.2


class DecisionEdge:
    """Represents an edge between nodes"""
    def __init__(self, from_node, to_node, label=''):
        self.from_node = from_node
        self.to_node = to_node
        self.label = label


# Define the tree structure with California Housing features
def build_decision_tree():
    """Build the complete decision tree structure"""
    nodes = {}
    
    # Start
    nodes['START'] = DecisionNode(5, 10, 'Feature j\nCompute M1·M2·M3\nVIF·Δ_interact', 
                                   'start', '#8b5cf6')
    
    # First decision
    nodes['M1'] = DecisionNode(5, 8, 'M1 (Univariate R²)\nsubstantial?', 
                                'decision', '#3b82f6')
    
    # Left branch (M1 high)
    nodes['VIF'] = DecisionNode(2.5, 6, 'VIF > 5?\nCollinear?', 
                                 'decision', '#3b82f6')
    nodes['M3A'] = DecisionNode(2.5, 4, 'M3 (Permutation)\nhigh?', 
                                 'decision', '#3b82f6')
    nodes['STRONG'] = DecisionNode(1.5, 2, '✅ STRONG\nIndependent\npredictor\n(MedInc)', 
                                    'outcome', '#10b981')
    nodes['ABSORBED'] = DecisionNode(3.5, 2, '⚠️ ABSORBED\nSignal displaced\nin joint model', 
                                      'outcome', '#f59e0b')
    nodes['COLLINEAR'] = DecisionNode(0.5, 4, '⚡ COLLINEAR\nUnstable weights\n(AveRooms)', 
                                       'outcome', '#f97316')
    
    # Right branch (M1 low)
    nodes['M2M3'] = DecisionNode(7.5, 6, 'M2 or M3 high\ndespite low M1?', 
                                  'decision', '#3b82f6')
    nodes['DROP'] = DecisionNode(9.5, 4, '❌ DROP\nNear-zero signal\n(Population)', 
                                  'outcome', '#ef4444')
    nodes['JOINT'] = DecisionNode(6.5, 4, 'Joint uplift\nΔ_interact > 0?', 
                                   'decision', '#3b82f6')
    nodes['IRREPLACEABLE'] = DecisionNode(5.5, 2, '✅ IRREPLACEABLE\nComplementary\n(Lat+Lon)', 
                                           'outcome', '#10b981')
    nodes['PROXY'] = DecisionNode(7.5, 2, '⚠️ PROXY\nBorrowed signal\nfrom partner', 
                                   'outcome', '#f59e0b')
    
    # Define edges
    edges = [
        DecisionEdge(nodes['START'], nodes['M1'], ''),
        DecisionEdge(nodes['M1'], nodes['VIF'], 'Yes\nStandalone'),
        DecisionEdge(nodes['M1'], nodes['M2M3'], 'No\nWeak alone'),
        DecisionEdge(nodes['VIF'], nodes['M3A'], 'No'),
        DecisionEdge(nodes['VIF'], nodes['COLLINEAR'], 'Yes'),
        DecisionEdge(nodes['M3A'], nodes['STRONG'], 'Yes'),
        DecisionEdge(nodes['M3A'], nodes['ABSORBED'], 'No'),
        DecisionEdge(nodes['M2M3'], nodes['DROP'], 'No'),
        DecisionEdge(nodes['M2M3'], nodes['JOINT'], 'Yes'),
        DecisionEdge(nodes['JOINT'], nodes['IRREPLACEABLE'], 'Yes'),
        DecisionEdge(nodes['JOINT'], nodes['PROXY'], 'No'),
    ]
    
    return nodes, edges


# Define example feature paths through the tree
FEATURE_PATHS = {
    'MedInc': {
        'path': ['START', 'M1', 'VIF', 'M3A', 'STRONG'],
        'color': '#34d399',
        'label': 'MedInc: M1=0.47, VIF=1.5, M3=0.33',
        'scores': 'M1 ✓ | VIF ✓ | M3 ✓'
    },
    'Lat+Lon': {
        'path': ['START', 'M1', 'M2M3', 'JOINT', 'IRREPLACEABLE'],
        'color': '#60a5fa',
        'label': 'Latitude+Longitude: M1≈0, M2/M3 high, Δ=+$10k',
        'scores': 'M1 ✗ | M2/M3 ✓ | Δ_interact ✓'
    },
    'AveRooms': {
        'path': ['START', 'M1', 'VIF', 'COLLINEAR'],
        'color': '#fb923c',
        'label': 'AveRooms: M1=0.02, VIF=7.2',
        'scores': 'M1 ~ | VIF ✗'
    },
    'Population': {
        'path': ['START', 'M1', 'M2M3', 'DROP'],
        'color': '#f87171',
        'label': 'Population: M1≈0, M2≈0, M3≈0',
        'scores': 'M1 ✗ | M2 ✗ | M3 ✗'
    }
}


def draw_tree(ax, nodes, edges, highlight_path=None, highlight_index=0, alpha=0.3):
    """Draw the decision tree with optional path highlighting"""
    ax.clear()
    ax.set_xlim(-1, 11)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Draw edges first (behind nodes)
    for edge in edges:
        x1, y1 = edge.from_node.x, edge.from_node.y
        x2, y2 = edge.to_node.x, edge.to_node.y
        
        # Check if this edge is in the highlighted path
        edge_active = False
        edge_color = '#4a5568'
        edge_width = 1.5
        edge_alpha = 0.3
        
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                if (edge.from_node in [nodes[highlight_path[i]]] and 
                    edge.to_node in [nodes[highlight_path[i+1]]]):
                    if i < highlight_index:
                        edge_active = True
                        edge_color = '#fbbf24'
                        edge_width = 3
                        edge_alpha = 1.0
                    break
        
        arrow = FancyArrowPatch(
            (x1, y1 - 0.5), (x2, y2 + 0.6),
            arrowstyle='->', mutation_scale=20,
            color=edge_color, linewidth=edge_width,
            alpha=edge_alpha, zorder=1
        )
        ax.add_patch(arrow)
        
        # Edge label
        if edge.label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, edge.label, fontsize=8,
                   ha='left', va='center', alpha=0.6)
    
    # Draw nodes
    for name, node in nodes.items():
        # Check if this node is in the highlighted path
        node_active = False
        node_alpha = alpha
        node_edgecolor = '#e2e8f0'
        node_linewidth = 1.5
        
        if highlight_path and name in highlight_path:
            path_idx = highlight_path.index(name)
            if path_idx <= highlight_index:
                node_active = True
                node_alpha = 1.0
                node_edgecolor = '#fbbf24'
                node_linewidth = 3
        
        # Choose box style based on node type
        if node.node_type == 'decision':
            boxstyle = 'round,pad=0.3'
        elif node.node_type == 'start':
            boxstyle = 'round,pad=0.4'
        else:
            boxstyle = 'round,pad=0.3'
        
        bbox = FancyBboxPatch(
            (node.x - node.width/2, node.y - node.height/2),
            node.width, node.height,
            boxstyle=boxstyle,
            facecolor=node.color,
            edgecolor=node_edgecolor,
            linewidth=node_linewidth,
            alpha=node_alpha,
            zorder=2
        )
        ax.add_patch(bbox)
        
        # Node text
        ax.text(node.x, node.y, node.text, fontsize=9,
               ha='center', va='center', alpha=node_alpha,
               zorder=3, weight='bold' if node_active else 'normal')


def generate_flow_animation():
    """Generate the slow-motion decision flow animation"""
    print("Generating ch03-feature-candidacy-flow.gif...")
    
    nodes, edges = build_decision_tree()
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='#1a1a2e')
    
    # Animation sequence:
    # 1. Show empty tree (30 frames)
    # 2. For each feature (4 features):
    #    - Show feature name/scores (20 frames)
    #    - Light up path step by step (15 frames per step)
    #    - Pause at outcome (40 frames)
    # 3. Show all paths together (50 frames)
    
    total_frames = 30 + 4 * (20 + 5 * 15 + 40) + 50
    
    feature_order = ['MedInc', 'Lat+Lon', 'AveRooms', 'Population']
    
    def update(frame):
        # Phase 1: Empty tree
        if frame < 30:
            draw_tree(ax, nodes, edges)
            ax.text(5, 11.5, 'Feature Candidacy Decision Flow', 
                   fontsize=16, ha='center', weight='bold')
            return
        
        frame -= 30
        
        # Phase 2: Each feature flows through
        feature_duration = 20 + 5 * 15 + 40  # intro + path + pause
        feature_idx = frame // feature_duration
        
        if feature_idx < 4:
            feature_name = feature_order[feature_idx]
            feature_data = FEATURE_PATHS[feature_name]
            local_frame = frame % feature_duration
            
            if local_frame < 20:
                # Show feature introduction
                draw_tree(ax, nodes, edges, alpha=0.2)
                ax.text(5, 11.5, f'Testing: {feature_data["label"]}',
                       fontsize=14, ha='center', weight='bold',
                       color=feature_data['color'])
                ax.text(5, 11, feature_data['scores'],
                       fontsize=11, ha='center', alpha=0.8)
            else:
                local_frame -= 20
                step_idx = min(local_frame // 15, len(feature_data['path']) - 1)
                
                draw_tree(ax, nodes, edges, 
                         highlight_path=feature_data['path'],
                         highlight_index=step_idx,
                         alpha=0.2)
                
                ax.text(5, 11.5, f'{feature_name}',
                       fontsize=14, ha='center', weight='bold',
                       color=feature_data['color'])
                
                # Show current step description
                current_node = nodes[feature_data['path'][step_idx]]
                if step_idx == len(feature_data['path']) - 1:
                    verdict = current_node.text.split('\n')[0]
                    ax.text(5, 11, f'Verdict: {verdict}',
                           fontsize=12, ha='center', weight='bold',
                           color=feature_data['color'])
            return
        
        # Phase 3: Show all paths together
        frame -= 4 * feature_duration
        
        ax.clear()
        ax.set_xlim(-1, 11)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Draw tree structure lightly
        for edge in edges:
            x1, y1 = edge.from_node.x, edge.from_node.y
            x2, y2 = edge.to_node.x, edge.to_node.y
            arrow = FancyArrowPatch(
                (x1, y1 - 0.5), (x2, y2 + 0.6),
                arrowstyle='->', mutation_scale=20,
                color='#4a5568', linewidth=1.5,
                alpha=0.15, zorder=1
            )
            ax.add_patch(arrow)
        
        for name, node in nodes.items():
            bbox = FancyBboxPatch(
                (node.x - node.width/2, node.y - node.height/2),
                node.width, node.height,
                boxstyle='round,pad=0.3',
                facecolor=node.color,
                edgecolor='#e2e8f0',
                linewidth=1.5,
                alpha=0.2,
                zorder=2
            )
            ax.add_patch(bbox)
            ax.text(node.x, node.y, node.text, fontsize=9,
                   ha='center', va='center', alpha=0.3, zorder=3)
        
        # Draw all feature paths as colored lines
        for feat_name, feat_data in FEATURE_PATHS.items():
            path = feat_data['path']
            color = feat_data['color']
            
            for i in range(len(path) - 1):
                n1 = nodes[path[i]]
                n2 = nodes[path[i + 1]]
                arrow = FancyArrowPatch(
                    (n1.x, n1.y - 0.5), (n2.x, n2.y + 0.6),
                    arrowstyle='->', mutation_scale=25,
                    color=color, linewidth=3,
                    alpha=0.7, zorder=5
                )
                ax.add_patch(arrow)
        
        ax.text(5, 11.5, 'All Features Diagnosed', 
               fontsize=16, ha='center', weight='bold')
        
        # Legend
        legend_y = 0.5
        for i, (feat_name, feat_data) in enumerate(FEATURE_PATHS.items()):
            ax.plot([], [], color=feat_data['color'], linewidth=3, 
                   label=feat_name, alpha=0.8)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    anim = FuncAnimation(fig, update, frames=total_frames, 
                        interval=100, repeat=True)
    
    save_path = 'notes/01-ml/01_regression/ch03_feature_importance/img/ch03-feature-candidacy-flow.gif'
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
    plt.close()
    
    print(f"✓ Saved {save_path}")
    print(f"  Duration: {total_frames/10:.1f} seconds")
    print(f"  Features shown: {', '.join(feature_order)}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Feature Candidacy Decision Flow Animation")
    print("="*60 + "\n")
    
    generate_flow_animation()
    
    print("\n" + "="*60)
    print("✓ Decision flow animation complete!")
    print("="*60 + "\n")
    
    print("Animation shows:")
    print("  1. Empty decision tree (3 sec)")
    print("  2. MedInc → Strong independent predictor")
    print("  3. Lat+Lon → Jointly irreplaceable")
    print("  4. AveRooms → Collinear signal (VIF > 5)")
    print("  5. Population → Drop candidate")
    print("  6. All paths overlaid (5 sec)")
    print("\nTotal duration: ~50 seconds (slow-motion for clarity)")
