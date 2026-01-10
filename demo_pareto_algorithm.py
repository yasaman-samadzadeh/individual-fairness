"""
Interactive Demo: How Pareto Front Algorithm Works

This script demonstrates step-by-step how the Pareto front is calculated
with a simple example, showing the dominance check process.
"""

import numpy as np
import matplotlib.pyplot as plt


def is_dominated(point, all_points):
    """
    Check if a point is dominated by any other point.
    
    A point is dominated if another point is better (lower) in ALL objectives.
    """
    for other in all_points:
        if np.array_equal(other, point):
            continue
        
        # Check if 'other' dominates 'point'
        # Dominance: other is better (lower) in ALL objectives
        if np.all(other <= point) and np.any(other < point):
            return True, other
    
    return False, None


def find_pareto_front_naive(points):
    """
    Naive implementation: Check each point for dominance.
    """
    pareto_front = []
    dominated_points = []
    
    print("=" * 70)
    print("PARETO FRONT CALCULATION - STEP BY STEP")
    print("=" * 70)
    
    for i, point in enumerate(points):
        print(f"\nChecking point {i+1}: error={point[0]:.3f}, inconsistency={point[1]:.3f}")
        is_dom, dominator = is_dominated(point, points)
        
        if is_dom:
            print(f"  ❌ DOMINATED by point with error={dominator[0]:.3f}, inconsistency={dominator[1]:.3f}")
            dominated_points.append((i, point, dominator))
        else:
            print(f"  ✅ PARETO OPTIMAL (not dominated)")
            pareto_front.append((i, point))
    
    return pareto_front, dominated_points


def visualize_pareto_demo():
    """Visualize Pareto front calculation with example points."""
    
    # Example: 10 configurations with (error, inconsistency) costs
    # Lower is better for both objectives
    configurations = np.array([
        [0.20, 0.10],  # Config 0: High error, low inconsistency
        [0.15, 0.15],  # Config 1: Medium error, medium inconsistency
        [0.18, 0.08],  # Config 2: Medium-high error, low inconsistency
        [0.12, 0.20],  # Config 3: Low error, high inconsistency
        [0.25, 0.12],  # Config 4: Very high error, medium inconsistency (DOMINATED)
        [0.16, 0.18],  # Config 5: Medium-low error, high inconsistency
        [0.10, 0.25],  # Config 6: Low error, very high inconsistency
        [0.22, 0.14],  # Config 7: High error, medium-high inconsistency (DOMINATED)
        [0.14, 0.12],  # Config 8: Medium error, low inconsistency
        [0.11, 0.22],  # Config 9: Low error, high inconsistency
    ])
    
    # Find Pareto front
    pareto_front, dominated = find_pareto_front_naive(configurations)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal configurations: {len(configurations)}")
    print(f"Pareto-optimal: {len(pareto_front)}")
    print(f"Dominated (removed): {len(dominated)}")
    
    print(f"\n✅ Pareto-optimal configurations:")
    pareto_points = []
    for idx, point in pareto_front:
        pareto_points.append(point)
        acc = 1 - point[0]
        cons = 1 - point[1]
        print(f"  Config {idx}: error={point[0]:.3f}, inconsistency={point[1]:.3f} → "
              f"accuracy={acc:.3f}, consistency={cons:.3f}")
    
    print(f"\n❌ Dominated configurations (removed from front):")
    for idx, point, dominator in dominated:
        acc = 1 - point[0]
        cons = 1 - point[1]
        print(f"  Config {idx}: error={point[0]:.3f}, inconsistency={point[1]:.3f} → "
              f"accuracy={acc:.3f}, consistency={cons:.3f}")
        print(f"      Dominated by: error={dominator[0]:.3f}, inconsistency={dominator[1]:.3f}")
    
    # Sort Pareto front by error (first objective) for plotting
    pareto_points = np.array(pareto_points)
    sort_idx = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sort_idx]
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cost space (error, inconsistency) - Lower is better
    ax1 = axes[0]
    
    # All points
    ax1.scatter(configurations[:, 0], configurations[:, 1],
               c='lightblue', s=100, alpha=0.6, label='All configurations', zorder=2)
    
    # Dominated points (highlighted)
    dominated_points = np.array([point for _, point, _ in dominated])
    if len(dominated_points) > 0:
        ax1.scatter(dominated_points[:, 0], dominated_points[:, 1],
                   c='red', s=150, alpha=0.8, marker='x', linewidths=3,
                   label='Dominated (removed)', zorder=3)
    
    # Pareto front
    ax1.scatter(pareto_points[:, 0], pareto_points[:, 1],
               c='green', s=200, edgecolors='black', linewidths=2,
               marker='o', label='Pareto Front', zorder=4)
    ax1.plot(pareto_points[:, 0], pareto_points[:, 1],
            'g--', linewidth=2, alpha=0.7, zorder=1)
    
    # Annotate points
    for i, (err, inc) in enumerate(configurations):
        ax1.annotate(f'{i}', (err, inc), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    ax1.set_xlabel('Error (1 - Balanced Accuracy)', fontsize=12)
    ax1.set_ylabel('Inconsistency (1 - Counterfactual Consistency)', fontsize=12)
    ax1.set_title('Pareto Front in Cost Space\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 0.27)
    ax1.set_ylim(0.05, 0.27)
    
    # Add arrows showing "better" directions
    ax1.annotate('', xy=(0.05, 0.07), xytext=(0.05, 0.25),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5))
    ax1.text(0.07, 0.16, 'Better\nFairness', rotation=90, fontsize=10, color='blue', alpha=0.7)
    
    ax1.annotate('', xy=(0.25, 0.07), xytext=(0.05, 0.07),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5))
    ax1.text(0.15, 0.04, 'Better Accuracy', fontsize=10, color='blue', alpha=0.7)
    
    # Plot 2: Score space (accuracy, consistency) - Higher is better
    ax2 = axes[1]
    
    # Convert costs to scores
    all_scores = 1 - configurations
    pareto_scores = 1 - pareto_points
    if len(dominated_points) > 0:
        dominated_scores = 1 - dominated_points
    
    # All points
    ax2.scatter(all_scores[:, 0], all_scores[:, 1],
               c='lightblue', s=100, alpha=0.6, label='All configurations', zorder=2)
    
    # Dominated points
    if len(dominated_points) > 0:
        ax2.scatter(dominated_scores[:, 0], dominated_scores[:, 1],
                   c='red', s=150, alpha=0.8, marker='x', linewidths=3,
                   label='Dominated (removed)', zorder=3)
    
    # Pareto front
    ax2.scatter(pareto_scores[:, 0], pareto_scores[:, 1],
               c='green', s=200, edgecolors='black', linewidths=2,
               marker='o', label='Pareto Front', zorder=4)
    ax2.plot(pareto_scores[:, 0], pareto_scores[:, 1],
            'g--', linewidth=2, alpha=0.7, zorder=1)
    
    # Annotate points
    for i, (acc, cons) in enumerate(all_scores):
        ax2.annotate(f'{i}', (acc, cons), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    ax2.set_xlabel('Balanced Accuracy', fontsize=12)
    ax2.set_ylabel('Counterfactual Consistency', fontsize=12)
    ax2.set_title('Pareto Front in Score Space\n(Higher = Better)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.73, 1.02)
    ax2.set_ylim(0.73, 0.98)
    
    # Add arrows showing "better" directions
    ax2.annotate('', xy=(0.75, 0.96), xytext=(0.75, 0.76),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5))
    ax2.text(0.77, 0.86, 'Better\nFairness', rotation=90, fontsize=10, color='blue', alpha=0.7)
    
    ax2.annotate('', xy=(0.98, 0.75), xytext=(0.76, 0.75),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5))
    ax2.text(0.87, 0.73, 'Better Accuracy', fontsize=10, color='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/pareto_algorithm_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n" + "=" * 70)
    print(f"Visualization saved to: plots/pareto_algorithm_demo.png")
    print("=" * 70)
    plt.close()


if __name__ == "__main__":
    visualize_pareto_demo()

