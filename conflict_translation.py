import contextily as ctx
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from typing import Dict


# ---------------------- CONFIGURATION CLASSES -------------------------------------------------------------------
# **************************************************************************************************************

class Distribution(Enum):
    """Force distribution types for spatial force application."""
    POINT = 'point'
    GAUSSIAN = 'gaussian'
    LINEAR = 'linear'
    CONSTANT = 'constant'


@dataclass
class ForceConfig:
    """
    Configuration parameters for force translation from conflict events.

    Attributes:
        base_magnitude (float): Base force magnitude before scaling
        fatality_weight (float): Weight factor for fatality scaling (α in combined linear-log scaling)
        civilian_factor (float): Multiplier for civilian targeting events (γ_c)
        decay_rate (float): Temporal decay rate constant (λ)
        distribution (Distribution): Spatial force distribution type
        base_radius (float): Base influence radius in meters
        expansion_rate (float): Rate of radius expansion over time (μ)
        max_expansion (float): Maximum radius expansion factor (β)
    """
    base_magnitude: float
    fatality_weight: float  # α in combined linear-log scaling
    civilian_factor: float  # γ_c for civilian targeting
    decay_rate: float  # λ for temporal decay
    distribution: Distribution
    base_radius: float
    expansion_rate: float  # μ for radius expansion
    max_expansion: float  # β for maximum radius increase


# ---------------------- EVENT TRANSLATION CLASS -----------------------------------------------------------------
# **************************************************************************************************************

class EventTranslator:
    """
    Translates conflict events into mechanical force configurations for FEA analysis.

    This class handles the conversion of conflict event data into force parameters
    suitable for finite element analysis, including magnitude calculation, spatial
    distribution, and temporal dynamics.
    """

    def __init__(self, democracy_index: float, event_type_config, subevent_type_config=None, democracy_adjustment=None):
        """
        Initialize event translator with configurations.

        Parameters:
            democracy_index (float): V-Dem liberal democracy index (0-1)
            event_type_config (Dict[str, ForceConfig]): Configuration for each event type
            subevent_type_config (Dict[str, ForceConfig], optional): Override configurations for specific sub-event types
            democracy_adjustment (Dict[str, Union[List[str], bool]], optional): Configuration for democracy index adjustment
        """
        self.democracy_index = democracy_index
        self.event_type_config = event_type_config
        self.subevent_type_config = subevent_type_config or {}
        self.democracy_adjustment = democracy_adjustment or {}

    def get_config(self, event_type: str, subevent_type: str) -> ForceConfig:
        """Get configuration, prioritizing sub-event type if available."""
        if subevent_type in self.subevent_type_config:
            return self.subevent_type_config[subevent_type]
        return self.event_type_config[event_type]

    def _should_apply_democracy_adjustment(self, event_type: str, subevent_type: str) -> bool:
        """Determine if democracy adjustment should be applied to the event."""
        if event_type not in self.democracy_adjustment:
            return False

        config = self.democracy_adjustment[event_type]
        apply_to = config.get('apply_to', ['event_type'])
        types = config.get('types', True)

        if 'event_type' in apply_to and (types is True or event_type in types):
            return True

        if 'subevent_type' in apply_to and (types is True or subevent_type in types):
            return True

        return False

    def calculate_force_magnitude(self, event_type: str, subevent_type: str, fatalities: int,
                                  civilian_targeting: bool, event_date: datetime, current_date: datetime) -> float:
        """
        Calculate total force magnitude for an event based on multiple factors.

        Parameters:
            event_type (str): Primary event classification
            subevent_type (str): Specific event subtype
            fatalities (int): Number of fatalities in the event
            civilian_targeting (bool): Whether event targeted civilians
            event_date (datetime): When the event occurred
            current_date (datetime): Current analysis date

        Returns:
            float: Calculated force magnitude in Newtons
        """
        config = self.get_config(event_type, subevent_type)
        base_magnitude = config.base_magnitude

        # Apply democracy adjustment if configured
        if self._should_apply_democracy_adjustment(event_type, subevent_type):
            base_magnitude *= (1 - self.democracy_index)

        # Calculate fatality scaling
        fatality_scale = (
                config.fatality_weight * (1 + fatalities) +
                (1 - config.fatality_weight) * (1 + np.log1p(fatalities))
        )

        # Apply civilian targeting factor
        intensity = fatality_scale * (1 + config.civilian_factor * civilian_targeting)

        # Calculate temporal decay
        days_passed = (current_date - event_date).days
        cutoff_days = np.log(100) / config.decay_rate

        if days_passed >= cutoff_days:
            return 0.0

        temporal_factor = np.exp(-config.decay_rate * days_passed)

        return base_magnitude * intensity * temporal_factor

    def calculate_force_radius(self, event_type: str, subevent_type: str, event_date: datetime,
                               current_date: datetime) -> float:
        """
        Calculate force radius including temporal expansion.

        Parameters:
            event_type (str): Primary event classification
            subevent_type (str): Specific event subtype
            event_date (datetime): When the event occurred
            current_date (datetime): Current analysis date

        Returns:
            float: Force influence radius in meters
        """
        config = self.get_config(event_type, subevent_type)

        # Calculate temporal expansion
        days_passed = (current_date - event_date).days
        expansion_factor = 1 + config.max_expansion * (
                1 - np.exp(-config.expansion_rate * days_passed)
        )

        return config.base_radius * expansion_factor

    def translate_events(self, events_df: pd.DataFrame, current_date: datetime) -> list:
        """
        Translate all events into force configurations.

        Parameters:
            events_df (pd.DataFrame): DataFrame containing conflict events
            current_date (datetime): Date for temporal calculations

        Returns:
            list: List of force configurations for the plate model
        """
        force_configs = []

        for _, event in events_df.iterrows():

            if event['event_date'] > current_date:
                continue

            # Calculate force magnitude
            magnitude = self.calculate_force_magnitude(
                event_type=event['event_type'],
                subevent_type=event['sub_event_type'],
                fatalities=event['fatalities'],
                civilian_targeting=event['civilian_targeting'],
                event_date=event['event_date'],
                current_date=current_date
            )

            # Skip if force has decayed completely
            if magnitude == 0:
                continue

            # Get distribution type and radius
            config = self.get_config(event['event_type'], event['sub_event_type'])

            # Create force configuration
            force_config = {
                'coordinates': (event['longitude'], event['latitude']),
                'magnitude': magnitude,
                'distribution': config.distribution.value
            }

            # Add radius if not point force
            if config.distribution != Distribution.POINT:
                radius = self.calculate_force_radius(
                    event_type=event['event_type'],
                    subevent_type=event['sub_event_type'],
                    event_date=event['event_date'],
                    current_date=current_date
                )

                force_config['radius'] = radius

            force_configs.append(force_config)

        return force_configs


# ---------------------- FORCE APPLICATION FUNCTIONS --------------------------------------------------------------
# **************************************************************************************************************

def apply_force_at_location(mesh, coordinates, force_magnitude, radius=None, distribution='gaussian', verbose=False):
    """
    Apply a directional force to simulate plate deformation at a specific location.

    Parameters:
        mesh (FEAMesh2D): Mesh object containing geometry and material properties
        coordinates (tuple): Physical coordinates (x, y) where force is applied
        force_magnitude (float): Force magnitude in Newtons
        radius (float, optional): Radius of influence for distributed load in meters
        distribution (str): Force distribution type ('point', 'gaussian', 'linear', 'constant')
        verbose (bool): Whether to print debug information

    Returns:
        tuple: (coordinates, force_vector) where coordinates is the center of force
               and force_vector is numpy array of length mesh.ndof with applied nodal forces
    """
    import numpy as np

    direction = (0, 1)  # Force direction: purely upward in y
    x, y = coordinates
    dx, dy = direction

    # Validate that the force location is within mesh bounds
    if not (mesh.bounds[0] <= x <= mesh.bounds[2] and
            mesh.bounds[1] <= y <= mesh.bounds[3]):
        raise ValueError(f"Coordinates {coordinates} outside mesh bounds {mesh.bounds}")

    # Case 1: Point force application
    if distribution == 'point' or radius is None:
        f = np.zeros(mesh.ndof)

        # Convert (x, y) to normalized [0..1] in each dimension
        x_norm = (x - mesh.bounds[0]) / (mesh.bounds[2] - mesh.bounds[0])
        y_norm = (y - mesh.bounds[1]) / (mesh.bounds[3] - mesh.bounds[1])

        # Find the nearest grid node in the mesh
        node_x = int(round(x_norm * mesh.nx))
        node_y = int(round(y_norm * mesh.ny))

        node_x = max(0, min(node_x, mesh.nx))
        node_y = max(0, min(node_y, mesh.ny))
        node = node_y * (mesh.nx + 1) + node_x

        if node >= mesh.nnodes:
            raise ValueError("Calculated node index exceeds mesh size")

        dof = mesh.dof_map[node, 0]
        f[dof] = force_magnitude

        if verbose:
            print(f"[Point Force] Applied at node {node}, DOF={dof}")
            print(f"Force magnitude: {force_magnitude:.3g} N")

        return (x, y), f

    # Case 2: Distributed load over an area
    x_coords = np.linspace(mesh.bounds[0], mesh.bounds[2], mesh.nx + 1)
    y_coords = np.linspace(mesh.bounds[1], mesh.bounds[3], mesh.ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords)

    radius_degrees = radius / 111000 if radius else None

    distances = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

    # Calculate force distribution weights based on distribution type
    if distribution == 'gaussian':
        if not radius_degrees:
            raise ValueError("Gaussian distribution requires a non-null radius.")
        sigma = radius_degrees / 3.0
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))

    elif distribution == 'linear':
        if not radius_degrees:
            raise ValueError("Linear distribution requires a non-null radius.")
        weights = np.maximum(0.0, 1.0 - distances / radius_degrees)

    elif distribution == 'constant':
        if not radius_degrees:
            raise ValueError("Constant distribution requires a non-null radius.")
        weights = (distances <= radius_degrees).astype(float)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    f = np.zeros(mesh.ndof)
    total_weight = np.sum(weights)
    if total_weight < 1e-15:
        if verbose:
            print("Warning: No nodes within the specified radius => no force applied.")
        return (x, y), f

    # Normalize weights and apply force
    weights /= total_weight

    for i in range(mesh.ny + 1):
        for j in range(mesh.nx + 1):
            w = weights[i, j]
            if w > 0.0:
                node = i * (mesh.nx + 1) + j
                dof = mesh.dof_map[node, 0]
                f[dof] = force_magnitude * w

    if verbose:
        print(f"[Distributed Force] distribution='{distribution}'")
        print(f"Center of load = ({x:.3f}, {y:.3f}), radius={radius_degrees:.3g} deg")
        print(f"Number of nodes loaded = {np.count_nonzero(weights > 0)}")
        print(f"Sum of all nodal forces = {np.sum(f):.3g} N (should ~ {force_magnitude:.3g} N)")

    return (x, y), f


def apply_multiple_forces(mesh, force_configs, fixed_nodes, verbose=False):
    """
    Apply multiple forces to the plate and solve using superposition.

    Parameters:
        mesh (FEAMesh2D): Mesh object with material properties
        force_configs (list): List of force configurations, each containing coordinates, magnitude, radius, and distribution
        fixed_nodes (list): DOF indices for boundary conditions
        verbose (bool): Whether to print progress information

    Returns:
        np.ndarray: Combined displacement solution vector
    """
    # Initialize total displacement vector
    u_total = np.zeros(mesh.ndof)

    if verbose:
        print(f"\nProcessing {len(force_configs)} force impacts:")

    # Process each force configuration
    for i, config in enumerate(force_configs, 1):
        if verbose:
            print(f"\nForce {i}:")
            print(f"Location: {config['coordinates']}")
            print(f"Magnitude: {config['magnitude']:.2e} N")

            if 'radius' in config:
                print(f"Influence radius: {config['radius']}")

        # Generate force distribution and get global force vector
        if config['magnitude'] > 0:
            force_location, force_vector = apply_force_at_location(
                mesh,
                coordinates=config['coordinates'],
                force_magnitude=config['magnitude'],
                radius=config.get('radius'),
                distribution=config.get('distribution'),
                verbose=verbose
            )

            # Solve and accumulate results using superposition
            u_total += mesh._solve(force_location, force_vector, fixed_nodes)

        print(f"Force {i}/{len(force_configs)} processed successfully")

    print("\nAll forces processed - solution complete")
    return u_total


# ---------------------- VISUALIZATION FUNCTIONS -----------------------------------------------------------------
# **************************************************************************************************************

def get_event_type_colors():
    """Define consistent colors for each event type for visualization."""
    return {
        'Battles': '#E41A1C',  # Red
        'Explosions/Remote violence': '#984EA3',  # Purple
        'Violence against civilians': '#FF7F00',  # Orange
        'Protests': '#4DAF4A',  # Green
        'Riots': '#377EB8',  # Blue
        'Strategic developments': '#FFFF33'  # Yellow
    }


def preview_multiple_forces(mesh, force_configs: list, events_df: pd.DataFrame):
    """
    Preview force locations and areas of application on a map.

    Parameters:
        mesh (FEAMesh2D): Mesh object containing geographic boundaries
        force_configs (list): List of force configurations to preview
        events_df (pd.DataFrame): DataFrame containing event information for labeling
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up basemap
    bounds = mesh.gdf_buffered.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    try:
        import contextily as ctx
        ctx.add_basemap(ax,
                        crs=mesh.region_crs,
                        source=ctx.providers.CartoDB.Positron,
                        zoom=6)
    except Exception as e:
        print(f"Warning: Could not add basemap: {str(e)}")

    # Plot boundaries
    mesh.gdf_original.boundary.plot(ax=ax, color='red', linewidth=1.5,
                                    label='Country Boundary')
    mesh.gdf_buffered.boundary.plot(ax=ax, color='blue', linewidth=1,
                                    linestyle='--', label='Analysis Region')

    # Create color mapping for event types
    event_colors = get_event_type_colors()

    # Handle any missing event types
    missing_types = set(events_df['event_type']) - set(event_colors)
    if missing_types:
        additional_colors = plt.cm.Set3(np.linspace(0, 1, len(missing_types)))
        event_colors.update(dict(zip(missing_types, additional_colors)))

    # Get time range for transparency gradient
    min_date = events_df['event_date'].min()
    max_date = events_df['event_date'].max()
    time_range = (max_date - min_date).days

    # Plot forces
    for i, (config, event) in enumerate(zip(force_configs, events_df.itertuples())):
        coords = config['coordinates']
        radius = config.get('radius')
        magnitude = config['magnitude']
        dist_type = config.get('distribution', 'point')

        # Calculate time-based transparency
        if time_range > 0:
            days_old = (max_date - event.event_date).days
            alpha = 1 - (days_old / time_range) * 0.8  # Keep minimum 0.2 opacity
        else:
            alpha = 1

        # Get base color for event type
        base_color = event_colors[event.event_type]

        # Create label
        if radius:
            label = f'{event.event_type}: {magnitude:.2e} N\n{dist_type}, r={radius / 1000:.1f} km'
        else:
            label = f'{event.event_type}: {magnitude:.2e} N\npoint force'

        # Plot force location
        ax.plot(coords[0], coords[1], 'o', color=base_color, markersize=8,
                alpha=alpha, label=label)

        if radius:
            circle = plt.Circle(coords, radius / 111000, color=base_color, alpha=alpha * 0.2)
            ax.add_patch(circle)

    plt.title('Force Locations and Areas of Application', pad=20, fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels,
                    title='Legend\n(opacity indicates time)',
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    fontsize=10)
    leg.get_title().set_fontsize(12)

    plt.tight_layout()
    plt.show()


def display_forces_summary(force_configs: list):
    """
    Display summary statistics of forces to be applied.

    Parameters:
        force_configs (list): List of force configurations
    """
    print(f"\nTotal forces to apply: {len(force_configs)}")
    print("\nForce summary:")
    for i, config in enumerate(force_configs, 1):
        print(f"\nForce {i}:")
        print(f"Location: {config['coordinates']}")
        print(f"Magnitude: {config['magnitude']:.2e} N")
        if 'radius' in config:
            print(f"Radius: {config['radius'] / 1000:.1f} km")
            print(f"Distribution: {config['distribution']}")


def visualize_conflict_events(mesh, events_df, current_date, translator, debug=False, legend=True,
                              additional_boundaries_file=None):
    """
    Visualize conflict events on a map with temporal dynamics and force calculations.

    Parameters:
        mesh (FEAMesh2D): Mesh object containing geographic boundaries
        events_df (pd.DataFrame): DataFrame containing conflict event data
        current_date (datetime): Current analysis date for temporal calculations
        translator (EventTranslator): Translator object for force calculations
        debug (bool): Whether to print debug information
        legend (bool): Whether to show legend
        additional_boundaries_file (str, optional): Path to additional boundary shapefile
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up basemap
    bounds = mesh.gdf_buffered.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    try:
        import contextily as ctx
        ctx.add_basemap(ax,
                        crs=mesh.region_crs,
                        source=ctx.providers.CartoDB.Positron,
                        zoom=6)
    except Exception as e:
        print(f"Warning: Could not add basemap: {str(e)}")

    # Plot boundaries
    mesh.gdf_original.boundary.plot(ax=ax, color='red', linewidth=1.5,
                                    label='Country Boundary')
    mesh.gdf_buffered.boundary.plot(ax=ax, color='blue', linewidth=1,
                                    linestyle='--', label='Analysis Region')

    if additional_boundaries_file:
        additional_boundaries = gpd.read_file(additional_boundaries_file)
        additional_boundaries.plot(ax=ax, color='gray', alpha=0.3, linewidth=1)

    # Create color mapping for event types
    color_map = get_event_type_colors()

    # Handle missing event types
    missing_types = set(events_df['event_type']) - set(color_map)
    if missing_types:
        additional_colors = cm.Set3(np.linspace(0, 1, len(missing_types)))
        color_map.update(dict(zip(missing_types, additional_colors)))

    if debug:
        print("\nStarting event processing:")
        print(f"Total events in DataFrame: {len(events_df)}")
        print(f"Current date: {current_date}")
        print("\nFirst few rows of the DataFrame:")
        print(events_df.head())
        print("\nDataFrame columns:", events_df.columns.tolist())

    # Initialize counters for processing summary
    events_plotted = 0
    events_skipped_future = 0
    events_skipped_decay = 0
    events_error = 0

    # Process and plot each event
    for idx, event in events_df.iterrows():
        try:
            # Convert event_date to datetime if needed
            if isinstance(event['event_date'], str):
                event_date = datetime.strptime(event['event_date'], '%Y-%m-%d')
            else:
                event_date = event['event_date']

            # Debug output for first few events
            if idx < 5 and debug:
                print(f"\nProcessing event {idx}:")
                print(f"Event date: {event_date}")
                print(f"Event type: {event['event_type']}")
                print(f"Location: ({event['longitude']}, {event['latitude']})")

            # Skip future events
            if event_date > current_date:
                events_skipped_future += 1
                continue

            # Calculate force properties
            magnitude = translator.calculate_force_magnitude(
                event_type=event['event_type'],
                subevent_type=event['sub_event_type'],
                fatalities=event['fatalities'],
                civilian_targeting=event.get('civilian_targeting', False),
                event_date=event_date,
                current_date=current_date
            )

            if magnitude == 0:
                events_skipped_decay += 1
                continue

            # Calculate radius
            radius_meters = translator.calculate_force_radius(
                event_type=event['event_type'],
                subevent_type=event['sub_event_type'],
                event_date=event_date,
                current_date=current_date
            )

            # Convert to degrees for visualization
            radius_degrees = (radius_meters / 111000)

            # Ensure minimum radius for visibility
            min_radius = 0.04  # Minimum radius in degrees
            if radius_degrees == 0:
                radius_degrees = min_radius

            # Calculate opacity based on temporal decay
            config = translator.get_config(event['event_type'], event['sub_event_type'])
            days_passed = (current_date - event_date).days
            opacity = np.exp(-config.decay_rate * days_passed)

            # Debug output
            if idx < 5 and debug:
                print(f"Magnitude: {magnitude}")
                print(f"Radius (meters): {radius_meters}")
                print(f"Radius (degrees): {radius_degrees}")
                print(f"Opacity: {opacity}")

            # Plot event visualization
            color = color_map.get(event['event_type'], '#999999')

            # Main influence circle
            circle = patches.Circle(
                (event['longitude'], event['latitude']),
                radius=radius_degrees,
                color=color,
                alpha=opacity * 0.6,
                fill=True,
                zorder=2
            )
            ax.add_patch(circle)

            # Center point (always visible)
            center = patches.Circle(
                (event['longitude'], event['latitude']),
                radius=min_radius / 2,  # Fixed size for center point
                color=color,
                alpha=min(opacity * 1.2, 1.0),
                fill=True,
                zorder=3
            )
            ax.add_patch(center)

            events_plotted += 1

        except Exception as e:
            events_error += 1
            if debug:
                print(f"Error processing event {idx}: {str(e)}")
                print(f"Event data: {event}")

    if debug:
        print("\nProcessing summary:")
        print(f"Events plotted: {events_plotted}")
        print(f"Events skipped (future): {events_skipped_future}")
        print(f"Events skipped (decay): {events_skipped_decay}")
        print(f"Events with errors: {events_error}")

    # Add title and create legend
    ax.set_title(
        f'Conflict Events Map: {events_df["event_date"].min().strftime("%Y-%m-%d")} - {current_date.strftime("%Y-%m-%d")}')

    if legend:
        legend_elements = [
            Line2D([], [],
                   marker='o',
                   color=color,
                   label=event_type,
                   linestyle=None
                   )
            for event_type, color in color_map.items() if event_type in events_df['event_type'].unique()
        ]

        boundary_legend_elements = [
            Line2D([], [], color='white'),
            Line2D([], [], color='red', linewidth=1.5, linestyle='-', label='Country Boundary'),
            Line2D([], [], color='blue', linewidth=1, linestyle='--', label='Analysis Region')
        ]

        legend_elements.extend(boundary_legend_elements)

        ax.legend(handles=legend_elements,
                  title='Conflict Event Types (opacity indicates time)',
                  loc='upper left',
                  bbox_to_anchor=(1.05, 1),
                  )

    plt.tight_layout()
    plt.show()