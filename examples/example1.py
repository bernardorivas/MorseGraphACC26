#!/usr/bin/env python3
"""Example 1: Toggle Switch

Computes Morse graphs, Morse sets, and regions of attraction
for the toggle switch using different methods.

Methods:
    1. F_integration (f): Numerical ODE integration of the ground-truth system
    2. F_data: Data-driven approach with "outside" mapping for unmapped boxes
    3. F_gaussianprocess (GP): GP-based outer approximation with confidence bounds (arXiv:2210.01292v1)
    4. F_Lipschitz (hat_f): Lipschitz-based bounds (hat_f = f for now)
    5. F_integration (hat_f): Numerical integration where hat_f = f (currently same as Method 1)
"""

import numpy as np
import os

from MorseGraph.grids import UniformGrid
from MorseGraph.dynamics import F_integration, F_Lipschitz, F_data, F_gaussianprocess
from MorseGraph.systems import SwitchingSystem
from MorseGraph.learning import train_gp_from_data
from MorseGraph.utils import generate_trajectory_data, define_tau_map
from MorseGraph.postprocessing import post_processing_example_1
from MorseGraph.analysis import full_morse_graph_analysis
from MorseGraph.plot import visualize_morse_sets_graph_basins


# ==============================================================================
# System Parameters
# ==============================================================================

# Production parameters
L1, T1, U1 = 1.0, 3.0, 5.0
L2, T2, U2 = 1.0, 3.0, 5.0

# Degradation rates
gamma1, gamma2 = 1.00, 1.00

# Integration time (tau)
tau = 1.0

# Data generation parameters
N_trajectories = 50     # Number of initial trajectories
T_total = 10*tau        # Total integration time for each trajectory
random_seed = 42        # For reproducibility
# Expected data pairs: N_trajectories × (T_total/tau)


# ==============================================================================
# System Definition
# ==============================================================================

# Define toggle switch as SwitchingSystem
# Polynomials: p1(x) = x1 - T1, p2(x) = x2 - T2
polynomials = [
    lambda x: x[0] - T1,
    lambda x: x[1] - T2,
]

# 4 modes based on sign patterns: [sign(p1), sign(p2)]
# Mode 0: [-, -] (both low)  → high production
# Mode 1: [-, +] (p1 low, p2 high)
# Mode 2: [+, -] (p1 high, p2 low)
# Mode 3: [+, +] (both high) → low production
vector_fields = [
    lambda x: np.array([-gamma1*x[0] + U1, -gamma2*x[1] + U2]),  # Mode 0
    lambda x: np.array([-gamma1*x[0] + U1, -gamma2*x[1] + L2]),  # Mode 1
    lambda x: np.array([-gamma1*x[0] + L1, -gamma2*x[1] + U2]),  # Mode 2
    lambda x: np.array([-gamma1*x[0] + L1, -gamma2*x[1] + L2]),  # Mode 3
]

ode_system = SwitchingSystem(polynomials, vector_fields)


# ==============================================================================
# MorseGraph Parameters
# ==============================================================================

# Output directory
output_dir = f'example_1_tau_{tau:.1f}'.replace('.', '_')

# Phase space parameters
dim = 2
lower_bounds = np.array([0.0, 0.0])
upper_bounds = np.array([6.0, 6.0])

# Grid resolution : subdivide n times in each direction, so 2^7 x 2^7 boxes
subdivision = 7

# Plot labels
xlabel = 'x1'
ylabel = 'x2'

# ==============================================================================
# Main Computation
# ==============================================================================

def main():
    """
    Computes Morse graphs for the toggle switch using 5 different F(rect) methods.

    Methods:
    1. F_integration (f): Numerical integration of the ground-truth system
    2. F_data: Data-driven approach with "outside" mapping for unmapped boxes
    3. F_gaussianprocess (GP): GP-based outer approximation with confidence bounds
    4. F_Lipschitz (hat_f): Lipschitz-based bounds
    5. F_integration (hat_f): Numerical integration where hat_f = f
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("MorseGraph Example 1: Toggle Switch")
    print("Comparing 5 different F(rect) computation methods")
    print("=" * 80 + "\n")

    # Set random seeds for reproducibility
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed set to {random_seed} for reproducibility\n")

    # Set up common UniformGrid for all methods
    divisions = np.array([2**subdivision, 2**subdivision], dtype=int)
    grid = UniformGrid(bounds=np.array([lower_bounds, upper_bounds]),
                       divisions=divisions)

    # Compute padding parameter: grid cell diameter
    box_size = (upper_bounds - lower_bounds) / (2**subdivision)
    epsilon = np.linalg.norm(box_size)

    print(f"Grid configuration:")
    print(f"  Resolution: {divisions[0]} × {divisions[1]} = {len(grid.get_boxes())} boxes")
    print(f"  Padding (epsilon): {epsilon:.6f}\n")

    results_summary = []

    # =========================================================================
    # METHOD 1: F_integration - Ground-Truth f
    # =========================================================================
    print("=" * 80)
    print("METHOD 1: F_integration - Numerical ODE Integration (Ground-truth f)")
    print("=" * 80)

    F1 = F_integration(ode_system, tau)

    # Compute Morse graph analysis
    box_map1, morse_graph1, basins1 = full_morse_graph_analysis(grid, F1)

    # Visualize
    _ = visualize_morse_sets_graph_basins(
        grid, morse_graph1, basins1, box_map1,
        'F_integration (f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_integration_f.png'),
        show_outside=False
    )

    # Create results dict
    results1 = {
        'method': 'F_integration (f)',
        'num_morse_sets': len(morse_graph1.nodes()),
        'num_edges': len(morse_graph1.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph1.nodes())},
        'basin_box_counts': {i: len(basins1[ms]) for i, ms in enumerate(morse_graph1.nodes())}
    }

    print(f"  Morse sets: {results1['num_morse_sets']}")
    print(f"  Edges: {results1['num_edges']}")

    combined_morse_graph, combined_basins = post_processing_example_1(
        grid, box_map1, morse_graph1, basins1, T1, T2
    )

    # Add combined metrics
    results1['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph.nodes())}
    results1['combined_basin_box_counts'] = {i: len(combined_basins[node]) for i, node in enumerate(combined_morse_graph.nodes())}
    results1['combined_morse_graph'] = combined_morse_graph
    results1['combined_basins'] = combined_basins

    results_summary.append(results1)

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph, combined_basins, combined_morse_graph,
        'F_integration (f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_integration_f_combined.png'),
        show_outside=False
    )

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    print("=" * 80)
    print("DATA GENERATION for F_data and F_gaussianprocess")
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
    mat_filename = os.path.join(output_dir, 'toggle_switch_data.mat')
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
    print(f"  Using {len(X)} trajectory samples from shared dataset...")

    # Create F_data with "outside" mapping
    F2 = F_data(X, Y, grid, map_empty='outside')

    # Compute Morse graph analysis
    box_map2, morse_graph2, basins2 = full_morse_graph_analysis(grid, F2)

    # Visualize
    _ = visualize_morse_sets_graph_basins(
        grid, morse_graph2, basins2, box_map2,
        'F_data',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_data.png'),
        show_outside=True
    )

    # Create results dict
    results2 = {
        'method': 'F_data',
        'num_morse_sets': len(morse_graph2.nodes()),
        'num_edges': len(morse_graph2.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph2.nodes())},
        'basin_box_counts': {i: len(basins2[ms]) for i, ms in enumerate(morse_graph2.nodes())}
    }

    print(f"  Morse sets: {results2['num_morse_sets']}")
    print(f"  Edges: {results2['num_edges']}")

    # Post-processing for METHOD 2
    combined_morse_graph2, combined_basins2 = post_processing_example_1(
        grid, box_map2, morse_graph2, basins2, T1, T2
    )

    # Add combined metrics
    results2['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph2.nodes())}
    results2['combined_basin_box_counts'] = {i: len(combined_basins2[node]) for i, node in enumerate(combined_morse_graph2.nodes())}
    results2['combined_morse_graph'] = combined_morse_graph2
    results2['combined_basins'] = combined_basins2

    results_summary.append(results2)

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph2, combined_basins2, combined_morse_graph2,
        'F_data - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_data_combined.png'),
        show_outside=False
    )

    # =========================================================================
    # METHOD 3: F_gaussianprocess (GP) - Data-Driven with Confidence Bounds
    # =========================================================================
    print("=" * 80)
    print("METHOD 3: F_gaussianprocess - GP-based Outer Approximation")
    print("=" * 80)
    print(f"  Using {len(X)} trajectory samples from dataset...")
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

    # Compute Morse graph analysis
    box_map3, morse_graph3, basins3 = full_morse_graph_analysis(grid, F3)

    # Visualize
    _ = visualize_morse_sets_graph_basins(
        grid, morse_graph3, basins3, box_map3,
        'F_gaussianprocess (GP)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_gaussianprocess.png'),
        show_outside=False
    )

    # Create results dict
    results3 = {
        'method': 'F_gaussianprocess (GP)',
        'num_morse_sets': len(morse_graph3.nodes()),
        'num_edges': len(morse_graph3.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph3.nodes())},
        'basin_box_counts': {i: len(basins3[ms]) for i, ms in enumerate(morse_graph3.nodes())}
    }

    print(f"  Morse sets: {results3['num_morse_sets']}")
    print(f"  Edges: {results3['num_edges']}\n")

    # Post-processing for METHOD 3
    combined_morse_graph3, combined_basins3 = post_processing_example_1(
        grid, box_map3, morse_graph3, basins3, T1, T2
    )

    # Add combined metrics
    results3['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph3.nodes())}
    results3['combined_basin_box_counts'] = {i: len(combined_basins3[node]) for i, node in enumerate(combined_morse_graph3.nodes())}
    results3['combined_morse_graph'] = combined_morse_graph3
    results3['combined_basins'] = combined_basins3

    results_summary.append(results3)

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph3, combined_basins3, combined_morse_graph3,
        'F_gaussianprocess (GP) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_gaussianprocess_combined.png'),
        show_outside=False
    )

    # =========================================================================
    # hat_f
    # =========================================================================

    hat_ode_system = ode_system

    # =========================================================================
    # METHOD 4: F_Lipschitz (hat_f)
    # =========================================================================

    print("=" * 80)
    print("METHOD 4: F_Lipschitz (hat_f) - Lipschitz-based Outer Approximation")
    print("=" * 80)

    # Compute bound for Lipschitz constant
    L_tau = np.exp(-min(gamma1, gamma2) * tau)
    print(f"  Lipschitz constant: L_tau = exp(-gamma_min * tau) = {L_tau:.6f}")

    # F_Lipschitz padding: L_tau * (box_diameter/2)
    box_diameter = np.linalg.norm(box_size)
    padding_radius = L_tau * box_diameter / 2
    print(f"  Box diameter (grid cell diagonal): {box_diameter:.6f}")
    print(f"  Padding radius: L_tau * (diameter/2) = {padding_radius:.6f}")

    # Define tau_map and F_Lipschitz
    hat_tau_map = define_tau_map(hat_ode_system, tau)
    F4 = F_Lipschitz(hat_tau_map, L_tau=L_tau, box_diameter=box_diameter)

    # Compute Morse graph analysis
    box_map4, morse_graph4, basins4 = full_morse_graph_analysis(grid, F4)

    # Visualize
    _ = visualize_morse_sets_graph_basins(
        grid, morse_graph4, basins4, box_map4,
        'F_Lipschitz (hat_f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_lipschitz_hatf.png'),
        show_outside=False
    )

    # Create results dict
    results4 = {
        'method': 'F_Lipschitz (hat_f)',
        'num_morse_sets': len(morse_graph4.nodes()),
        'num_edges': len(morse_graph4.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph4.nodes())},
        'basin_box_counts': {i: len(basins4[ms]) for i, ms in enumerate(morse_graph4.nodes())}
    }

    print(f"  Morse sets: {results4['num_morse_sets']}")
    print(f"  Edges: {results4['num_edges']}")

    # Post-processing for METHOD 4
    combined_morse_graph4, combined_basins4 = post_processing_example_1(
        grid, box_map4, morse_graph4, basins4, T1, T2
    )

    # Add combined metrics
    results4['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph4.nodes())}
    results4['combined_basin_box_counts'] = {i: len(combined_basins4[node]) for i, node in enumerate(combined_morse_graph4.nodes())}
    results4['combined_morse_graph'] = combined_morse_graph4
    results4['combined_basins'] = combined_basins4

    results_summary.append(results4)

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph4, combined_basins4, combined_morse_graph4,
        'F_Lipschitz (hat_f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_lipschitz_hatf_combined.png'),
        show_outside=False
    )

    # =========================================================================
    # METHOD 5: F_integration (hat_f)
    # =========================================================================
    print("=" * 80)
    print("METHOD 5: F_integration (hat_f) - Numerical Integration")
    print("=" * 80)
    print(f"  Note: hat_f = f (until specific hat_f is implemented)")

    F5 = F_integration(hat_ode_system, tau)

    # Compute Morse graph analysis
    box_map5, morse_graph5, basins5 = full_morse_graph_analysis(grid, F5)

    # Visualize
    _ = visualize_morse_sets_graph_basins(
        grid, morse_graph5, basins5, box_map5,
        'F_integration (hat_f)',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_integration_hatf.png'),
        show_outside=False
    )

    # Create results dict
    results5 = {
        'method': 'F_integration (hat_f)',
        'num_morse_sets': len(morse_graph5.nodes()),
        'num_edges': len(morse_graph5.edges()),
        'morse_set_box_counts': {i: len(ms) for i, ms in enumerate(morse_graph5.nodes())},
        'basin_box_counts': {i: len(basins5[ms]) for i, ms in enumerate(morse_graph5.nodes())}
    }

    print(f"  Morse sets: {results5['num_morse_sets']}")
    print(f"  Edges: {results5['num_edges']}")

    # Post-processing for METHOD 5
    combined_morse_graph5, combined_basins5 = post_processing_example_1(
        grid, box_map5, morse_graph5, basins5, T1, T2
    )

    # Add combined metrics
    results5['combined_morse_set_box_counts'] = {i: len(node) for i, node in enumerate(combined_morse_graph5.nodes())}
    results5['combined_basin_box_counts'] = {i: len(combined_basins5[node]) for i, node in enumerate(combined_morse_graph5.nodes())}
    results5['combined_morse_graph'] = combined_morse_graph5
    results5['combined_basins'] = combined_basins5

    results_summary.append(results5)

    # Visualize combined morse graph
    _ = visualize_morse_sets_graph_basins(
        grid, combined_morse_graph5, combined_basins5, combined_morse_graph5,
        'F_integration (hat_f) - Combined',
        np.array([lower_bounds, upper_bounds]),
        (xlabel, ylabel),
        os.path.join(output_dir, 'toggle_switch_integration_hatf_combined.png'),
        show_outside=False
    )

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("=" * 80)
    print("COMPARISON SUMMARY - BEFORE POST-PROCESSING")
    print("=" * 80)
    print(f"{'Method':<30} {'MS':<6} {'Edges':<8} {'Avg MS Size':<12} {'Avg Basin':<12}")
    print("-" * 80)
    for res in results_summary:
        avg_ms_size = np.mean(list(res['morse_set_box_counts'].values()))
        avg_basin = np.mean(list(res['basin_box_counts'].values()))
        print(f"{res['method']:<30} {res['num_morse_sets']:<6} {res['num_edges']:<8} "
              f"{avg_ms_size:<12.1f} {avg_basin:<12.1f}")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - AFTER POST-PROCESSING")
    print("=" * 80)
    print(f"{'Method':<30} {'MS':<6} {'Avg MS Size':<12} {'Avg Basin':<12}")
    print("-" * 80)
    for res in results_summary:
        num_collapsed = len(res['combined_morse_set_box_counts'])
        avg_collapsed_ms = np.mean(list(res['combined_morse_set_box_counts'].values()))
        avg_collapsed_basin = np.mean(list(res['combined_basin_box_counts'].values()))
        print(f"{res['method']:<30} {num_collapsed:<6} {avg_collapsed_ms:<12.1f} {avg_collapsed_basin:<12.1f}")
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