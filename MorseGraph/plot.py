import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from typing import Dict, Set, FrozenSet

from .grids import AbstractGrid

def plot_morse_sets(grid: AbstractGrid, morse_graph: nx.DiGraph, ax: plt.Axes = None,
                   box_map: nx.DiGraph = None, show_outside: bool = False, **kwargs):
    """
    Plots the Morse sets on a grid.

    :param grid: The grid object used for the computation.
    :param morse_graph: The Morse graph (NetworkX DiGraph) containing the Morse sets as nodes.
                       Each node should have a 'color' attribute (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param box_map: The BoxMap (directed graph) from compute_box_map(). Required if show_outside=True.
    :param show_outside: If True, paint boxes that map outside the domain in grey. Requires box_map.
    :param kwargs: Additional keyword arguments to pass to the PatchCollection.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Extract morse sets from the NetworkX graph
    morse_sets = morse_graph.nodes()

    # Get all boxes from the grid
    all_boxes = grid.get_boxes()

    rects = []
    colors = []

    # Use colors from node attributes (assigned by compute_morse_graph)
    for morse_set in morse_sets:
        # Get color from node attribute, fallback to viridis if not present
        if 'color' in morse_graph.nodes[morse_set]:
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Fallback for backward compatibility (colorblind-friendly)
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('viridis')
            color = cmap(list(morse_sets).index(morse_set) / max(num_sets, 10))

        for box_index in morse_set:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1])
                rects.append(rect)
                colors.append(color)

    if rects:
        pc = PatchCollection(rects, facecolors=colors, alpha=0.7, **kwargs)
        ax.add_collection(pc)

    # Optionally show boxes that map outside the domain (have data but no transitions)
    if show_outside and box_map is not None:
        # Collect all boxes in Morse sets
        morse_set_boxes = set()
        for morse_set in morse_sets:
            morse_set_boxes.update(morse_set)

        # Find boxes with out_degree == 0 that are not in any Morse set
        # These boxes have data but map outside the domain
        outside_boxes = set()
        for node in box_map.nodes():
            if box_map.out_degree(node) == 0 and node not in morse_set_boxes:
                outside_boxes.add(node)

        # Paint outside boxes grey
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor='grey',
                               edgecolor='none',
                               alpha=0.4)
                ax.add_patch(rect)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')

def plot_basins_of_attraction(grid: AbstractGrid, basins,
                             morse_graph: nx.DiGraph = None, ax: plt.Axes = None,
                             show_outside: bool = False, **kwargs):
    """
    Plots the basins of attraction with colors matching the Morse sets.

    :param grid: The grid object used for the computation.
    :param basins: Dict[FrozenSet[int], Set[int]]: box-level basins (from compute_all_morse_set_basins)
    :param morse_graph: The Morse graph with color attributes. If provided, uses colors from node attributes.
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param show_outside: If True, paint boxes not in any basin black (boxes mapping outside domain).
    :param kwargs: Additional keyword arguments to pass to Rectangle patches.
    """
    if ax is None:
        _, ax = plt.subplots()

    all_boxes = grid.get_boxes()

    # Plot each basin
    for attractor, basin in basins.items():
        # Get color from morse_graph node attributes if available
        if morse_graph and 'color' in morse_graph.nodes[attractor]:
            color = morse_graph.nodes[attractor]['color']
        else:
            # Fallback to viridis colormap for backward compatibility
            colors_cmap = plt.cm.get_cmap('viridis', len(basins))
            color = colors_cmap(list(basins.keys()).index(attractor))

        # Box-level basins: basin is a set of box indices
        # Plot basin boxes with lower opacity
        for box_index in basin:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=0.3, **kwargs)
                ax.add_patch(rect)

        # Plot attractor itself with full opacity
        for box_index in attractor:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor=color,
                               edgecolor='none',
                               alpha=1.0, **kwargs)
                ax.add_patch(rect)

    # Optionally show boxes that don't belong to any basin (map outside domain)
    if show_outside:
        all_box_indices = set(range(len(all_boxes)))
        # For box basins, union all basin box sets
        basin_box_indices = set().union(*basins.values())
        outside_boxes = all_box_indices - basin_box_indices

        # Paint outside boxes black
        for box_index in outside_boxes:
            if box_index < len(all_boxes):
                box = all_boxes[box_index]
                rect = Rectangle((box[0, 0], box[0, 1]),
                               box[1, 0] - box[0, 0],
                               box[1, 1] - box[0, 1],
                               facecolor='black',
                               edgecolor='none',
                               alpha=0.5)
                ax.add_patch(rect)

    ax.set_xlim(grid.bounds[0, 0], grid.bounds[1, 0])
    ax.set_ylim(grid.bounds[0, 1], grid.bounds[1, 1])
    ax.set_aspect('equal', adjustable='box')

def _hierarchical_dag_layout(G):
    """
    Compute a hierarchical layout for a directed acyclic graph (DAG).

    Nodes are arranged in layers based on their longest path to a source node.
    Nodes with no incoming edges are placed at the top; sinks at the bottom.

    :param G: A networkx DiGraph (should be acyclic)
    :return: A dictionary mapping nodes to (x, y) positions
    """
    # Compute longest path from each node to any source (node with in-degree 0)
    # This determines the "level" or "layer" of each node

    # For DAGs, we can use a topological sort and compute levels
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        # Not a DAG, return empty to trigger fallback
        raise ValueError("Graph is not a DAG")

    # Compute the level of each node (distance from sources)
    levels = {}
    for node in topo_order:
        # Find maximum level of predecessors
        predecessors = list(G.predecessors(node))
        if not predecessors:
            # Source node
            levels[node] = 0
        else:
            # Level is max level of predecessors + 1
            levels[node] = max(levels[pred] for pred in predecessors) + 1

    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Create positions: y-coordinate is level (inverted so 0 is at top)
    # x-coordinate spreads nodes horizontally within each level
    pos = {}
    max_level = max(levels.values()) if levels else 0

    for level in sorted(level_groups.keys()):
        nodes_in_level = level_groups[level]
        num_nodes = len(nodes_in_level)

        # Spread nodes horizontally
        for i, node in enumerate(nodes_in_level):
            if num_nodes == 1:
                x = 0
            else:
                x = (i - (num_nodes - 1) / 2) * 2

            # y goes from top (max_level) to bottom (0)
            y = max_level - level
            pos[node] = (x, y)

    return pos

def plot_morse_graph(morse_graph: nx.DiGraph, ax: plt.Axes = None,
                    morse_sets_colors: dict = None, node_size: int = 300,
                    arrowsize: int = 20, font_size: int = 8, show_box_counts: bool = False):
    """
    Plots the Morse graph with hierarchical layout.

    The graph is displayed as a directed acyclic graph (DAG) where nodes at the top
    represent sources (recurrent sets with no incoming edges from other recurrent sets)
    and nodes at the bottom represent sinks. This visualizes the partial order on
    recurrent components.

    :param morse_graph: The Morse graph to plot. Each node should have a 'color' attribute
                       (assigned by compute_morse_graph).
    :param ax: The matplotlib axes to plot on. If None, a new figure and axes are created.
    :param morse_sets_colors: Deprecated parameter, ignored. Colors are taken from node attributes.
    :param node_size: Size of the nodes.
    :param arrowsize: Size of the arrow heads.
    :param font_size: Font size for node labels.
    :param show_box_counts: If True, display the number of boxes in each Morse set next to the node number.
    """
    if ax is None:
        _, ax = plt.subplots()

    morse_sets = list(morse_graph.nodes())

    # Use colors from node attributes or generate
    node_colors = []
    for morse_set in morse_sets:
        color = None
        if 'color' in morse_graph.nodes[morse_set]:
            # Use color from node attribute
            color = morse_graph.nodes[morse_set]['color']
        else:
            # Generate color for backward compatibility (colorblind-friendly)
            num_sets = len(morse_sets)
            cmap = cm.get_cmap('viridis')
            color = cmap(morse_sets.index(morse_set) / max(num_sets, 10))

        # Convert numpy floats to python floats to avoid pygraphviz warning
        if hasattr(color, '__iter__'):
            color = tuple(float(c) for c in color)

        node_colors.append(color)
        # Update graph attribute for pygraphviz
        morse_graph.nodes[morse_set]['color'] = color

    # Create a mapping from frozenset to a shorter string representation
    if show_box_counts:
        node_labels = {node: f"{i+1}\n({len(node)})"
                      for i, node in enumerate(morse_sets)}
    else:
        node_labels = {node: str(i+1) for i, node in enumerate(morse_sets)}

    # Try hierarchical layout, with smart DAG-aware fallback
    try:
        from networkx.drawing.nx_agraph import pygraphviz_layout
        pos = pygraphviz_layout(morse_graph, prog='dot')
    except (ImportError, Exception):
        # Fallback: use a topological sort-based hierarchical layout
        try:
            pos = _hierarchical_dag_layout(morse_graph)
        except:
            # Final fallback: spring layout
            pos = nx.spring_layout(morse_graph, seed=42)

    # Center the layout by shifting positions so centroid is at origin
    if pos:
        pos_array = np.array(list(pos.values()))
        centroid = pos_array.mean(axis=0)
        pos = {node: (x - centroid[0], y - centroid[1]) for node, (x, y) in pos.items()}

    # Draw the graph components
    # Note: node_colors are RGBA tuples which matplotlib handles correctly
    nx.draw_networkx_nodes(morse_graph, pos, node_color=node_colors,
                          node_size=node_size, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(morse_graph, pos, edge_color='gray',
                          arrows=True, arrowsize=arrowsize, ax=ax, alpha=0.6,
                          connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(morse_graph, pos, labels=node_labels,
                           font_size=font_size, ax=ax)

    ax.set_title("Morse Graph (DAG of Recurrent Components)")
    ax.axis('off')

def visualize_morse_sets_graph_basins(grid, morse_graph, basins, box_map,
                                       method_label, bounds, labels, output_path,
                                       show_outside=False):
    """
    Create 3-panel visualization: Morse Sets | Morse Graph | Basins of Attraction.

    This is a pure visualization function that creates the standard 3-panel figure
    for Morse graph analysis. Separated from computation for modularity.

    :param grid: UniformGrid object
    :param morse_graph: NetworkX DiGraph of morse sets
    :param basins: Dict mapping morse sets (frozensets) to their basins (sets of box indices)
    :param box_map: NetworkX DiGraph of box-to-box transitions (for outside detection)
    :param method_label: Display label for plots (e.g., 'F_integration (f)')
    :param bounds: Domain bounds array of shape (2, D) [[mins], [maxs]]
    :param labels: Tuple of (xlabel, ylabel) for axis labels
    :param output_path: Full path to save the figure (e.g., './toggle_switch_integration_f.png')
    :param show_outside: Whether to highlight boxes that map outside (for F_data)
    :return: output_path

    Example:
        >>> from MorseGraph.analysis import compute_morse_graph_analysis
        >>> from MorseGraph.plot import visualize_morse_sets_graph_basins
        >>> box_map, morse_graph, basins = compute_morse_graph_analysis(grid, F)
        >>> visualize_morse_sets_graph_basins(
        ...     grid, morse_graph, basins, box_map,
        ...     'F_integration', bounds, ('x1', 'x2'), './output.png'
        ... )
    """
    import matplotlib.pyplot as plt
    import os

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    xlabel, ylabel = labels
    lower_bounds, upper_bounds = bounds[0], bounds[1]

    # Panel 1: Morse Sets
    plot_morse_sets(grid, morse_graph, ax=axes[0],
                    box_map=box_map if show_outside else None,
                    show_outside=show_outside)
    axes[0].set_xlim(lower_bounds[0], upper_bounds[0])
    axes[0].set_ylim(lower_bounds[1], upper_bounds[1])
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'Morse Sets - {method_label}')

    # Panel 2: Morse Graph
    plot_morse_graph(morse_graph, ax=axes[1])
    axes[1].set_title(f'Morse Graph - {method_label}')

    # Panel 3: Basins of Attraction
    plot_basins_of_attraction(grid, basins, morse_graph=morse_graph, ax=axes[2],
                              show_outside=show_outside)
    axes[2].set_xlim(lower_bounds[0], upper_bounds[0])
    axes[2].set_ylim(lower_bounds[1], upper_bounds[1])
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Basins of Attraction - {method_label}')

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path
