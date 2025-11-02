#!/usr/bin/env python3
"""Example 2: Piecewise Van der Pol Oscillator

Computes Morse graphs, Morse sets, and regions of attraction
for a piecewise Van der Pol-like ODE system with nonlinear force.
"""

import numpy as np
import os

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import F_integration, F_Lipschitz, F_data, F_gaussianprocess
from MorseGraph.systems import SwitchingSystem
from MorseGraph.learning import train_gp_from_data
from MorseGraph.utils import generate_trajectory_data, define_tau_map
from MorseGraph.postprocessing import post_processing_example_2
from MorseGraph.analysis import full_morse_graph_analysis
from MorseGraph.plot import visualize_morse_sets_graph_basins


# ==============================================================================
# System Parameters
# ==============================================================================

# Van der Pol parameter
mu = 1.0  # Damping coefficient

# Integration time (tau)
tau = 1.0

# Data generation parameters (for F_data and F_gaussianprocess)
N_trajectories = 50     # Number of initial trajectories
T_total = 10*tau        # Total integration time for each trajectory
random_seed = 42        # For reproducibility
# Expected data pairs: N_trajectories × (T_total/tau)


# ==============================================================================
# System Definition
# ==============================================================================

# Define piecewise Van der Pol oscillator as SwitchingSystem
# Polynomial: p(x) = x1^2 - 1
# - p(x) < 0 when |x1| < 1 → Mode 0 (cubic force)
# - p(x) > 0 when |x1| > 1 → Mode 1 (linear force)
polynomials = [
    lambda x: x[0]**2 - 1,
]

# 2 modes based on sign of polynomial
vector_fields = [
    # Mode 0: |x1| < 1 (cubic force)
    lambda x: np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]**3]),
    # Mode 1: |x1| > 1 (linear force)
    lambda x: np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]]),
]

ode_system = SwitchingSystem(polynomials, vector_fields)


# ==============================================================================
# MorseGraph Parameters
# ==============================================================================

# Output directory (includes tau value)
output_dir = f'example_2_tau_{tau:.1f}'.replace('.', '_')

# Phase space parameters
dim = 2
lower_bounds = np.array([-3.0, -3.0])
upper_bounds = np.array([3.0, 3.0])

# Grid resolution: 2^k intervals per dimension
grid_resolution = 7

# Plot labels
xlabel = 'x1'
ylabel = 'x2'


# ==============================================================================
# Main Computation
# ==============================================================================

def main():
    """
    Computes Morse graphs for the piecewise Van der Pol oscillator using 5 different F(rect) methods.

    All methods use the same dynamics (f = piecewise_vdp, hat_f = f):
    1. F_integration (f): Numerical ODE integration of the ground-truth system
    2. F_data: Data-driven approach with "outside" mapping for unmapped boxes
    3. F_gaussianprocess (GP): GP-based outer approximation with confidence bounds
    4. F_Lipschitz (hat_f): Lipschitz-based bounds (hat_f = f)
    5. F_integration (hat_f): Numerical integration (hat_f = f, same as Method 1)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("MorseGraph Example 2: Piecewise Van der Pol Oscillator")
    print("Comparing 5 different F(rect) computation methods")
    print("=" * 80 + "\n")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed set to {random_seed} for reproducibility\n")

    # Setup grid
    divisions = np.array([2**grid_resolution, 2**grid_resolution], dtype=int)
    grid = UniformGrid(bounds=np.array([lower_bounds, upper_bounds]),
                       divisions=divisions)

    # Compute padding parameter
    diameter = (upper_bounds - lower_bounds) / (2**grid_resolution)
    epsilon = np.linalg.norm(diameter)/2

    print(f"Grid configuration:")
    print(f"  Resolution: {divisions[0]} × {divisions[1]} = {len(grid.get_boxes())} boxes")
    print(f"  Padding (epsilon): {epsilon:.6f}")
    print(f"  Domain: [{lower_bounds[0]:.1f}, {upper_bounds[0]:.1f}] × [{lower_bounds[1]:.1f}, {upper_bounds[1]:.1f}]")

    # Define Wong colorblind-safe palette for morse set visualization
    wong_colors = [
        (0.0, 0.45, 0.70),   # Blue
        (0.90, 0.60, 0.0),   # Orange
        (0.0, 0.62, 0.45),   # Bluish green
        (0.95, 0.90, 0.25),  # Yellow
        (0.0, 0.45, 0.70),   # Sky blue
        (0.80, 0.40, 0.0),   # Vermillion
        (0.80, 0.60, 0.70),  # Reddish purple
        (0.0, 0.0, 0.0)      # Black
    ]

    def apply_wong_colors(morse_graph):
        """Apply Wong colorblind-safe palette to morse sets."""
        morse_sets = list(morse_graph.nodes())
        for i, morse_set in enumerate(morse_sets):
            color = wong_colors[i % len(wong_colors)]
            morse_graph.nodes[morse_set]['color'] = color

    results_summary = []

    # =========================================================================
    # METHOD 1: F_integration (f) - Ground Truth
    # =========================================================================
    print("=" * 80)
    print("METHOD 1: F_integration (f) - Numerical ODE Integration (Ground Truth)")
    print("=" * 80)

    F1 = F_integration(ode_system, tau)

    # Compute Morse graph analysis
    box_map1, morse_graph1, basins1 = full_morse_graph_analysis(grid, F1)

    # Apply Wong colorblind-safe palette
    apply_wong_colors(morse_graph1)

    # Visualize
    figure_path1 = visualize_morse_sets_graph_basins(
        grid, morse_graph1, basins1, box_map1,
        'F_integration (f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_integration_f.png'),
        show_outside=False
    )

    # Compute detailed metrics
    morse_set_box_counts = {ms: len(ms) for ms in morse_graph1.nodes()}
    basin_box_counts = {ms: len(basin) for ms, basin in basins1.items()}

    # Post-processing
    print("\nPost-processing morse graph...")
    combined_morse_graph, combined_basins = post_processing_example_2(
        grid, box_map1, morse_graph1, basins1
    )

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph, combined_basins, combined_morse_graph,
        'F_integration (f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_integration_f_combined.png'),
        show_outside=False
    )

    # Compute combined metrics
    combined_morse_set_box_counts = {ms: len(ms) for ms in combined_morse_graph.nodes()}
    combined_basin_box_counts = {ms: len(basin) for ms, basin in combined_basins.items()}

    # Create results dict
    results1 = {
        'method': 'F_integration (f)',
        'num_morse_sets': len(morse_graph1.nodes()),
        'num_edges': len(morse_graph1.edges()),
        'num_basins': len(basins1),
        'figure_path': figure_path1,
        'morse_set_box_counts': morse_set_box_counts,
        'basin_box_counts': basin_box_counts,
        'combined_morse_set_box_counts': combined_morse_set_box_counts,
        'combined_basin_box_counts': combined_basin_box_counts,
        'combined_morse_graph': combined_morse_graph,
        'combined_basins': combined_basins
    }
    print(f"  Morse sets: {results1['num_morse_sets']}")
    print(f"  Edges: {results1['num_edges']}")
    print(f"  Saved: {results1['figure_path']}\n")
    results_summary.append(results1)

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    print("=" * 80)
    print("DATA GENERATION")
    print("=" * 80)
    print(f"  Sampling along {N_trajectories} trajectories with T_total={T_total}, tau={tau}")
    print(f"  Expected: ~{N_trajectories * int(T_total/tau)} data pairs from {N_trajectories} integrations")

    # Generate trajectory data
    trajectories = generate_trajectory_data(ode_system, tau, N_trajectories,
                                           np.array([lower_bounds, upper_bounds]),
                                           T_total=T_total, seed=random_seed)
    print(f"  Generated {len(trajectories)} trajectories")

    # Extract tau-map pairs: X[i] = φ(i*τ, x₀), Y[i] = φ((i+1)*τ, x₀)
    X = np.vstack([traj[:-1] for traj in trajectories])  # First n-1 points from each trajectory
    Y = np.vstack([traj[1:] for traj in trajectories])   # Last n-1 points from each trajectory
    print(f"  Extracted X, Y with shapes {X.shape}, {Y.shape}")

    # Flatten all trajectory points for vector field evaluation
    trajectory_data = np.vstack(trajectories)  # All points: φ(0,x), φ(τ,x), ..., φ(T_total,x)
    print(f"  Computing vector field at {len(trajectory_data)} trajectory points...")
    F_field = np.array([ode_system(0, x) for x in trajectory_data])
    print(f"  Computed vector field at {len(trajectory_data)} points")

    # Save MATLAB-compatible data
    import scipy.io as sio
    mat_filename = os.path.join(output_dir, 'piecewise_vdp_data.mat')
    sio.savemat(mat_filename, {
        'trajectory_data': trajectory_data,  # All trajectory points
        'F_field': F_field           # Vector field f(x) at all trajectory points
    })
    print(f"  Saved data to: {mat_filename}")
    print(f"    - trajectory_data, F_field: vector field ({len(trajectory_data)} points)")

    # =========================================================================
    # METHOD 2: F_data - Data-Driven
    # =========================================================================
    print("=" * 80)
    print("METHOD 2: F_data - Data-Driven")
    print("=" * 80)
    print(f"  Using {len(X)} trajectory samples from dataset...")

    # Create F_data
    F2 = F_data(X, Y, grid, map_empty='outside')
    box_map2, morse_graph2, basins2 = full_morse_graph_analysis(grid, F2)

    # Apply Wong colorblind-safe palette
    apply_wong_colors(morse_graph2)

    figure_path2 = visualize_morse_sets_graph_basins(
        grid, morse_graph2, basins2, box_map2,
        'F_data',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_data.png'),
        show_outside=True
    )

    # Compute detailed metrics
    morse_set_box_counts = {ms: len(ms) for ms in morse_graph2.nodes()}
    basin_box_counts = {ms: len(basin) for ms, basin in basins2.items()}

    # Post-processing
    print("\nPost-processing morse graph...")
    combined_morse_graph2, combined_basins2 = post_processing_example_2(
        grid, box_map2, morse_graph2, basins2
    )

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph2, combined_basins2, combined_morse_graph2,
        'F_data - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_data_combined.png'),
        show_outside=False
    )

    # Compute combined metrics
    combined_morse_set_box_counts = {ms: len(ms) for ms in combined_morse_graph2.nodes()}
    combined_basin_box_counts = {ms: len(basin) for ms, basin in combined_basins2.items()}

    results2 = {
        'method': 'F_data',
        'num_morse_sets': len(morse_graph2.nodes()),
        'num_edges': len(morse_graph2.edges()),
        'num_basins': len(basins2),
        'figure_path': figure_path2,
        'morse_set_box_counts': morse_set_box_counts,
        'basin_box_counts': basin_box_counts,
        'combined_morse_set_box_counts': combined_morse_set_box_counts,
        'combined_basin_box_counts': combined_basin_box_counts,
        'combined_morse_graph': combined_morse_graph2,
        'combined_basins': combined_basins2
    }
    print(f"  Morse sets: {results2['num_morse_sets']}")
    print(f"  Edges: {results2['num_edges']}")
    print(f"  Saved: {results2['figure_path']}\n")
    results_summary.append(results2)

    # =========================================================================
    # METHOD 3: F_gaussianprocess (GP) - Data-Driven with Confidence Bounds
    # =========================================================================
    print("=" * 80)
    print("METHOD 3: F_gaussianprocess - GP-based Outer Approximation")
    print("=" * 80)
    print(f"  Using {len(X)} trajectory samples from shared dataset...")
    print(f"  Training GP model...")

    # Train GP on the trajectory data (same X, Y as METHOD 2)
    import time
    gp_start = time.time()
    gp_model = train_gp_from_data(X, Y, kernel_type='matern', nu=2.5)
    gp_train_time = time.time() - gp_start
    print(f"  GP training completed in {gp_train_time:.2f}s")
    print(f"  Confidence level: 95% (1-δ = 0.95)")

    # Create F_gaussianprocess with trained model
    F3 = F_gaussianprocess(gp_model, confidence_level=0.95, epsilon=0.0)
    box_map3, morse_graph3, basins3 = full_morse_graph_analysis(grid, F3)

    # Apply Wong colorblind-safe palette
    apply_wong_colors(morse_graph3)

    figure_path3 = visualize_morse_sets_graph_basins(
        grid, morse_graph3, basins3, box_map3,
        'F_gaussianprocess (GP)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_gaussianprocess.png'),
        show_outside=False
    )

    # Compute detailed metrics
    morse_set_box_counts = {ms: len(ms) for ms in morse_graph3.nodes()}
    basin_box_counts = {ms: len(basin) for ms, basin in basins3.items()}

    # Post-processing
    print("\nPost-processing morse graph...")
    combined_morse_graph3, combined_basins3 = post_processing_example_2(
        grid, box_map3, morse_graph3, basins3
    )

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph3, combined_basins3, combined_morse_graph3,
        'F_gaussianprocess (GP) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_gaussianprocess_combined.png'),
        show_outside=False
    )

    # Compute combined metrics
    combined_morse_set_box_counts = {ms: len(ms) for ms in combined_morse_graph3.nodes()}
    combined_basin_box_counts = {ms: len(basin) for ms, basin in combined_basins3.items()}

    results3 = {
        'method': 'F_gaussianprocess (GP)',
        'num_morse_sets': len(morse_graph3.nodes()),
        'num_edges': len(morse_graph3.edges()),
        'num_basins': len(basins3),
        'figure_path': figure_path3,
        'morse_set_box_counts': morse_set_box_counts,
        'basin_box_counts': basin_box_counts,
        'combined_morse_set_box_counts': combined_morse_set_box_counts,
        'combined_basin_box_counts': combined_basin_box_counts,
        'combined_morse_graph': combined_morse_graph3,
        'combined_basins': combined_basins3
    }
    print(f"  Morse sets: {results3['num_morse_sets']}")
    print(f"  Edges: {results3['num_edges']}")
    print(f"  Saved: {results3['figure_path']}")
    print(f"  Note: Uses same {len(X)} sparse samples as F_data")
    results_summary.append(results3)

    # =========================================================================
    # METHODS 4-5: Using same system (hat_f = f)
    # =========================================================================

    # =========================================================================
    # METHOD 4: F_Lipschitz (hat_f) - Lipschitz-based Outer Approximation
    # =========================================================================
    print("=" * 80)
    print("METHOD 4: F_Lipschitz (hat_f) - Lipschitz-based Outer Approximation")
    print("=" * 80)
    print("  Using hat_f = f (piecewise Van der Pol)")

    # Use Lipschitz constant for Van der Pol
    L_tau = 20.64
    print(f"  Lipschitz constant: L_tau = {L_tau:.6f}")

    # F_Lipschitz Padding: padding = L_tau * (box_diameter/2)
    box_diameter = np.linalg.norm(diameter)  # Diagonal of a grid cell
    padding_radius = L_tau * box_diameter / 2
    print(f"  Padding radius: L_tau * (diameter/2) = {padding_radius:.6f}")
    print(f"  Switching surface: |x1| = 1")

    # Create point map and F_Lipschitz using ode_system
    tau_map = define_tau_map(ode_system, tau)
    F4 = F_Lipschitz(tau_map, L_tau=L_tau, box_diameter=box_diameter, epsilon=0.0)
    box_map4, morse_graph4, basins4 = full_morse_graph_analysis(grid, F4)

    # Apply Wong colorblind-safe palette
    apply_wong_colors(morse_graph4)

    figure_path4 = visualize_morse_sets_graph_basins(
        grid, morse_graph4, basins4, box_map4,
        'F_Lipschitz (hat_f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_lipschitz_hatf.png'),
        show_outside=False
    )

    # Compute detailed metrics
    morse_set_box_counts = {ms: len(ms) for ms in morse_graph4.nodes()}
    basin_box_counts = {ms: len(basin) for ms, basin in basins4.items()}

    # Post-processing
    print("\nPost-processing morse graph...")
    combined_morse_graph4, combined_basins4 = post_processing_example_2(
        grid, box_map4, morse_graph4, basins4
    )

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph4, combined_basins4, combined_morse_graph4,
        'F_Lipschitz (hat_f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_lipschitz_hatf_combined.png'),
        show_outside=False
    )

    # Compute combined metrics
    combined_morse_set_box_counts = {ms: len(ms) for ms in combined_morse_graph4.nodes()}
    combined_basin_box_counts = {ms: len(basin) for ms, basin in combined_basins4.items()}

    results4 = {
        'method': 'F_Lipschitz (hat_f)',
        'num_morse_sets': len(morse_graph4.nodes()),
        'num_edges': len(morse_graph4.edges()),
        'num_basins': len(basins4),
        'figure_path': figure_path4,
        'morse_set_box_counts': morse_set_box_counts,
        'basin_box_counts': basin_box_counts,
        'combined_morse_set_box_counts': combined_morse_set_box_counts,
        'combined_basin_box_counts': combined_basin_box_counts,
        'combined_morse_graph': combined_morse_graph4,
        'combined_basins': combined_basins4
    }
    print(f"  Morse sets: {results4['num_morse_sets']}")
    print(f"  Edges: {results4['num_edges']}")
    print(f"  Saved: {results4['figure_path']}\n")
    results_summary.append(results4)

    # =========================================================================
    # METHOD 5: F_integration (hat_f) - Numerical Integration
    # =========================================================================
    print("=" * 80)
    print("METHOD 5: F_integration (hat_f) - Numerical Integration")
    print("=" * 80)
    print(f"  Using hat_f = f (piecewise Van der Pol)")
    print(f"  Note: This is identical to METHOD 1 since hat_f = f")

    F5 = F_integration(ode_system, tau)
    box_map5, morse_graph5, basins5 = full_morse_graph_analysis(grid, F5)

    # Apply Wong colorblind-safe palette
    apply_wong_colors(morse_graph5)

    figure_path5 = visualize_morse_sets_graph_basins(
        grid, morse_graph5, basins5, box_map5,
        'F_integration (hat_f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_integration_hatf.png'),
        show_outside=False
    )

    # Compute detailed metrics
    morse_set_box_counts = {ms: len(ms) for ms in morse_graph5.nodes()}
    basin_box_counts = {ms: len(basin) for ms, basin in basins5.items()}

    # Post-processing
    print("\nPost-processing morse graph...")
    combined_morse_graph5, combined_basins5 = post_processing_example_2(
        grid, box_map5, morse_graph5, basins5
    )

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph5, combined_basins5, combined_morse_graph5,
        'F_integration (hat_f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'piecewise_vdp_integration_hatf_combined.png'),
        show_outside=False
    )

    # Compute combined metrics
    combined_morse_set_box_counts = {ms: len(ms) for ms in combined_morse_graph5.nodes()}
    combined_basin_box_counts = {ms: len(basin) for ms, basin in combined_basins5.items()}

    results5 = {
        'method': 'F_integration (hat_f)',
        'num_morse_sets': len(morse_graph5.nodes()),
        'num_edges': len(morse_graph5.edges()),
        'num_basins': len(basins5),
        'figure_path': figure_path5,
        'morse_set_box_counts': morse_set_box_counts,
        'basin_box_counts': basin_box_counts,
        'combined_morse_set_box_counts': combined_morse_set_box_counts,
        'combined_basin_box_counts': combined_basin_box_counts,
        'combined_morse_graph': combined_morse_graph5,
        'combined_basins': combined_basins5
    }
    print(f"  Morse sets: {results5['num_morse_sets']}")
    print(f"  Edges: {results5['num_edges']}")
    print(f"  Saved: {results5['figure_path']}\n")
    results_summary.append(results5)

    # =========================================================================
    # Comparison Summary (Before Post-Processing)
    # =========================================================================
    print("=" * 80)
    print("COMPARISON SUMMARY (Before Post-Processing)")
    print("=" * 80)
    print(f"{'Method':<25} {'Morse Sets':<15} {'Edges':<10} {'Avg MS Size':<15} {'Avg Basin Size':<15}")
    print("-" * 90)
    for res in results_summary:
        # Compute averages
        avg_ms_size = np.mean(list(res['morse_set_box_counts'].values())) if res['morse_set_box_counts'] else 0
        avg_basin_size = np.mean(list(res['basin_box_counts'].values())) if res['basin_box_counts'] else 0
        print(f"{res['method']:<25} {res['num_morse_sets']:<15} {res['num_edges']:<10} {avg_ms_size:<15.1f} {avg_basin_size:<15.1f}")
    print()

    # =========================================================================
    # Comparison Summary (After Post-Processing)
    # =========================================================================
    print("=" * 80)
    print("COMPARISON SUMMARY (After Post-Processing)")
    print("=" * 80)
    print(f"{'Method':<25} {'Combined MS':<15} {'Avg MS Size':<15} {'Avg Basin Size':<15}")
    print("-" * 80)
    for res in results_summary:
        # Compute averages for combined results
        num_combined_ms = len(res['combined_morse_set_box_counts'])
        avg_combined_ms_size = np.mean(list(res['combined_morse_set_box_counts'].values())) if res['combined_morse_set_box_counts'] else 0
        avg_combined_basin_size = np.mean(list(res['combined_basin_box_counts'].values())) if res['combined_basin_box_counts'] else 0
        print(f"{res['method']:<25} {num_combined_ms:<15} {avg_combined_ms_size:<15.1f} {avg_combined_basin_size:<15.1f}")
    print()

    # =========================================================================
    # METRICS COMPARISON (vs Ground Truth)
    # =========================================================================
    from MorseGraph.metrics import compute_all_metrics

    print("\n" + "=" * 80)
    print("METRICS COMPARISON (vs Ground Truth - F_integration)")
    print("=" * 80)

    # Method 1 (F_integration) is Ground Truth
    gt_morse_graph = results_summary[0]['combined_morse_graph']
    gt_basins = results_summary[0]['combined_basins']

    metrics_table = []
    for i, result in enumerate(results_summary):
        if i == 0:  # Skip GT vs GT
            continue

        print(f"\nComputing metrics for {result['method']}...")
        metrics = compute_all_metrics(
            result['combined_morse_graph'],
            result['combined_basins'],
            gt_morse_graph,
            gt_basins,
            grid
        )

        metrics_table.append({
            'method': result['method'],
            'mean_iou': metrics['roa']['mean_iou'],
            'num_matched': metrics['roa']['num_matched'],
            'isomorphic': metrics['graph']['isomorphic'],
            'edit_distance': metrics['graph']['edit_distance'],
            'normalized_edit_distance': metrics['graph']['normalized_edit_distance']
        })

    # Print and save metrics table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Prepare table for both printing and saving
    table_lines = []
    table_lines.append("=" * 80)
    table_lines.append("METRICS COMPARISON (vs Ground Truth - F_integration)")
    table_lines.append("=" * 80)
    table_lines.append(f"{'Method':<30} {'IoU':<10} {'Matched':<10} {'Iso':<8} {'Edit Dst':<10}")
    table_lines.append("-" * 80)

    for m in metrics_table:
        iso_str = 'Yes' if m['isomorphic'] else 'No'
        edit_str = f"{m['normalized_edit_distance']:.3f}" if m['normalized_edit_distance'] is not None else 'N/A'
        line = f"{m['method']:<30} {m['mean_iou']:<10.4f} {m['num_matched']:<10} {iso_str:<8} {edit_str:<10}"
        table_lines.append(line)
        print(line)

    table_lines.append("=" * 80)
    print()

    # Save metrics table to .txt file
    txt_output_path = os.path.join(output_dir, 'metrics_comparison.txt')
    with open(txt_output_path, 'w') as f:
        f.write('\n'.join(table_lines))

    # Save metrics to JSON
    import json
    json_output_path = os.path.join(output_dir, 'metrics_comparison.json')
    with open(json_output_path, 'w') as f:
        json.dump(metrics_table, f, indent=2, default=str)

    print(f"Metrics saved to: {txt_output_path}")
    print(f"JSON data saved to: {json_output_path}\n")


if __name__ == "__main__":
    main()
