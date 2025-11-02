"""
Utility functions for data generation and spatial operations.

This module provides helper functions for working with trajectory data
and spatial operations on Morse sets.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import warnings
from typing import Callable, Any

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Tau-Map Data Generation
# =============================================================================

def generate_trajectory_data(ode_func, tau, N_trajectories, bounds, T_total=10, seed=None):
    """
    Generate trajectory data by sampling along ODE trajectories.

    Strategy: Sample N_trajectories initial conditions and integrate each for T_total time
    at time steps 0, tau, 2*tau, ..., T_total. Returns complete trajectories that can be
    used to extract tau-map pairs and vector field samples.

    This gives N_trajectories trajectories, each with (T_total/tau + 1) time points,
    providing efficient data generation for GP training and data-driven methods.

    :param ode_func: ODE function (t, x) -> dx/dt
    :param tau: Time step for τ-map
    :param N_trajectories: Number of initial trajectories
    :param bounds: Domain bounds array of shape (2, D)
    :param T_total: Total integration time (default: 10)
    :param seed: Random seed for reproducibility
    :return: List of trajectories, where each trajectory is array of shape (n_steps, dim)
             containing [φ(0,x), φ(τ,x), φ(2τ,x), ..., φ(T_total,x)]

    Example:
        >>> from MorseGraph.utils import generate_trajectory_data
        >>> from MorseGraph.systems import toggle_switch
        >>> from functools import partial
        >>> ode = partial(toggle_switch, L1=1, T1=3, U1=5, L2=1, T2=3, U2=5, gamma1=1, gamma2=1)
        >>> bounds = np.array([[0., 0.], [6., 6.]])
        >>> trajectories = generate_trajectory_data(ode, tau=1.0, N_trajectories=50, bounds=bounds, T_total=10)
        >>> print(f"Generated {len(trajectories)} trajectories, each with {len(trajectories[0])} points")
        >>> # Extract tau-map pairs: X = first n-1 points, Y = last n-1 points
        >>> X = np.vstack([traj[:-1] for traj in trajectories])
        >>> Y = np.vstack([traj[1:] for traj in trajectories])
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample N_trajectories random initial conditions
    dim = bounds.shape[1]
    X0 = np.random.uniform(bounds[0], bounds[1], size=(N_trajectories, dim))

    # Time points to sample along each trajectory
    # Use linspace to avoid floating-point errors with arange
    n_steps = int(np.round(T_total / tau)) + 1
    t_eval = np.linspace(0, T_total, n_steps)

    trajectories = []
    failed_count = 0

    # Integrate each trajectory
    for x0 in X0:
        try:
            sol = solve_ivp(ode_func, [0, T_total], x0, method='RK45',
                          t_eval=t_eval, dense_output=False)

            # Store complete trajectory: φ(0,x), φ(τ,x), ..., φ(T_total,x)
            trajectory = sol.y.T  # Shape: (n_steps, dim)
            trajectories.append(trajectory)

        except Exception as e:
            failed_count += 1
            print(f"Warning: Integration failed for point {x0}: {e}")
            continue

    # Report summary
    if failed_count > 0:
        failure_rate = failed_count / N_trajectories
        print(f"  Skipped {failed_count}/{N_trajectories} trajectories ({failure_rate*100:.1f}%) due to integration failures")
        if failure_rate > 0.1:
            warnings.warn(f"Over 10% integration failures; results may be biased")

    return trajectories


# =============================================================================
# Tau-Map Construction
# =============================================================================

def define_tau_map(ode_f, tau: float, method: str = 'RK45',
                         rtol: float = 1e-6, atol: float = 1e-8,
                         max_step: float = None):
    """
    Create a point map for time-τ integration using scipy.integrate.solve_ivp.

    Useful for creating the map_f argument for F_Lipschitz.

    :param ode_f: ODE right-hand side function f(t, x)
    :param tau: Integration time
    :param method: Integration method ('RK45', 'RK23', 'BDF', etc.)
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :param max_step: Maximum step size (for handling discontinuities)
    :return: map_f function that maps x -> φ_τ(x)

    Example:
        >>> def ode_rhs(t, x):
        ...     return np.array([-x[0] + 1, -x[1] + 2])
        >>> map_f = create_tau_map_scipy(ode_rhs, tau=0.1)
        >>> x_next = map_f(np.array([0.5, 0.5]))
    """
    def map_f(x):
        """Evaluate φ_τ(x) via ODE integration."""
        kwargs = dict(method=method, rtol=rtol, atol=atol)
        if max_step is not None:
            kwargs['max_step'] = max_step

        sol = solve_ivp(ode_f, [0.0, tau], x, **kwargs)

        if not sol.status == 0:
            warnings.warn(f"ODE integration warning: {sol.message}")

        return sol.y[:, -1]

    return map_f


# =============================================================================
# Morse Set Spatial Operations
# =============================================================================

def find_box_containing_point(grid, point):
    """
    Find the box index containing the given point.

    :param grid: UniformGrid object
    :param point: Point coordinates (numpy array or list)
    :return: Box index (int)
    """
    # For UniformGrid, compute box indices from point coordinates
    lower_bounds = grid.bounds[0]
    box_widths = (grid.bounds[1] - grid.bounds[0]) / grid.divisions

    # Compute which box the point falls into
    indices = np.floor((np.array(point) - lower_bounds) / box_widths).astype(int)

    # Clamp to valid range
    indices = np.clip(indices, 0, grid.divisions - 1)

    # Convert multi-dimensional index to linear box index
    # For 2D: box_idx = i * divisions[1] + j
    box_idx = indices[0] * grid.divisions[1] + indices[1]

    return box_idx


def compute_morse_set_centroid(grid, morse_set):
    """
    Compute the centroid of all boxes in a morse set.

    :param grid: UniformGrid object
    :param morse_set: Frozenset of box indices
    :return: Centroid coordinates (numpy array)
    """
    box_centers = []
    for box_idx in morse_set:
        # Convert box index to multi-dimensional indices
        indices = np.unravel_index(box_idx, grid.divisions)

        # Compute box center
        lower_bounds = grid.bounds[0]
        box_widths = (grid.bounds[1] - grid.bounds[0]) / grid.divisions
        center = lower_bounds + (np.array(indices) + 0.5) * box_widths
        box_centers.append(center)

    # Compute centroid as mean of all box centers
    return np.mean(box_centers, axis=0)
