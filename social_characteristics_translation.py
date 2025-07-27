import numpy as np
from typing import Optional, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
import warnings
from matplotlib import pyplot as plt
import rasterio
from rasterio.enums import Resampling


# ---------------------- DATA IMPORT AND NORMALIZATION FUNCTIONS ------------------------------------------------
# **************************************************************************************************************

def import_and_normalize_tiff(mesh, tiff_path: str, indicator_type: str = 'both', mid_point: float = None,
                              resampling_method: str = 'bilinear', fill_nan: float = 0.0, visualize: bool = True,
                              flip_data: bool = False):
    """
    Import a TIFF file, resample it to mesh dimensions, and normalize the data.

    Parameters:
        mesh (FEAMesh2D): Mesh object containing geometry and projection information
        tiff_path (str): Path to the TIFF file
        indicator_type (str): Data interpretation type
            - 'v' for vulnerability (-1 to 0)
            - 'r' for resilience (0 to 1)
            - 'both' for both ranges (-1 to 1)
            - 'p' for points (0 to 1) as prerequisite
        mid_point (float, optional): Value used as the midpoint for normalization when indicator_type is 'both'
        resampling_method (str): Method for resampling ('bilinear', 'nearest', 'average', etc.)
        fill_nan (float): Value to use for areas without data
        visualize (bool): Whether to create visualization plots
        flip_data (bool): Whether to flip the sign of the data before normalization

    Returns:
        tuple: (normalized_data, data_range) where normalized_data is normalized according to indicator type
               and data_range is original (min, max) values of the data
    """
    if indicator_type not in ['v', 'r', 'both', 'p']:
        raise ValueError("indicator_type must be 'v', 'r', 'both', or 'p'")

    if indicator_type == 'both' and mid_point is None:
        raise ValueError("mid_point must be provided when indicator_type is 'both'")

    # Treat prerequisite as resilience to achieve positive range mapping
    prerequisite_flag = False
    if indicator_type == 'p':
        prerequisite_flag = True
        indicator_type = 'r'

    try:
        with rasterio.open(tiff_path) as src:
            print(f"\nProcessing TIFF: {tiff_path}")
            print(f"Source Details:")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Original shape: {src.shape}")

            # Get mesh rectangle bounds for resampling
            mesh_bounds = mesh.bounds  # [xmin, ymin, xmax, ymax]

            # Create destination array matching mesh dimensions
            resampled_data = np.zeros((mesh.ny, mesh.nx))

            # Get proper transform for the mesh grid
            dst_transform = rasterio.transform.from_bounds(
                mesh_bounds[0], mesh_bounds[1],  # west, south
                mesh_bounds[2], mesh_bounds[3],  # east, north
                mesh.nx, mesh.ny
            )

            # Reproject data to mesh grid
            rasterio.warp.reproject(
                source=src.read(1),
                destination=resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=mesh.region_crs,
                resampling=getattr(Resampling, resampling_method),
                src_nodata=src.nodata,
                dst_nodata=fill_nan
            )

            # Handle nodata values and apply scaling
            resampled_data = np.where(np.isnan(resampled_data), fill_nan, resampled_data)
            if src.nodata is not None:
                resampled_data = np.where(resampled_data == src.nodata, fill_nan, resampled_data)

            # Flip data if requested
            if flip_data:
                valid_mask = resampled_data != fill_nan
                resampled_data[valid_mask] = -resampled_data[valid_mask]

            # Flip the data to match correct orientation
            resampled_data = np.flipud(resampled_data)

            # Calculate data range using valid data
            valid_data = resampled_data[resampled_data != fill_nan]
            if len(valid_data) == 0:
                print("Error: No valid data found in mesh area!")
                return None, None

            data_min, data_max = np.min(valid_data), np.max(valid_data)
            data_range = (data_min, data_max)
            print(f"\nValid data range: [{data_min:.6f}, {data_max:.6f}]")

            # Normalize based on indicator type
            normalized_data = np.zeros_like(resampled_data)
            if indicator_type == 'v':  # Vulnerability only (-1 to 0)
                normalized_data = -(resampled_data - data_min) / (data_max - data_min)  # Higher values -> more negative
            elif indicator_type == 'r':  # Resilience only (0 to 1)
                normalized_data = (resampled_data - data_min) / (data_max - data_min)
            else:  # Both (-1 to 1)
                print(f"Midpoint: {mid_point}")
                mask_below = resampled_data < mid_point
                mask_above = resampled_data >= mid_point

                if np.any(mask_below):
                    normalized_data[mask_below] = -1 + (
                            (resampled_data[mask_below] - data_min) /
                            (mid_point - data_min)
                    )
                if np.any(mask_above):
                    normalized_data[mask_above] = (
                            (resampled_data[mask_above] - mid_point) /
                            (data_max - mid_point)
                    )

            # Ensure NaN areas are filled
            normalized_data = np.where(resampled_data == fill_nan, fill_nan, normalized_data)

            if visualize:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                plt.suptitle(f"{tiff_path.split('.')[0]}", y=1.05, fontsize=14, fontweight='bold')

                # Plot resampled data
                extent = [mesh.bounds[0], mesh.bounds[2],
                          mesh.bounds[1], mesh.bounds[3]]
                im1 = axes[0].imshow(resampled_data, extent=extent,
                                     cmap='viridis', origin='lower')
                axes[0].set_title(f'Resampled Data\nRange: [{data_min:.2f}, {data_max:.2f}]')

                # Add original boundary
                boundary = mesh.gdf_original.boundary
                if mesh.region_crs != mesh.gdf_original.crs:
                    boundary = boundary.to_crs(mesh.region_crs)
                boundary.plot(ax=axes[0], color='red', linewidth=1)

                axes[0].set_xlabel('Longitude')
                axes[0].set_ylabel('Latitude')
                plt.colorbar(im1, ax=axes[0])

                # Plot data coverage
                coverage = np.where(resampled_data != fill_nan, 1, 0)
                im2 = axes[1].imshow(coverage, extent=extent,
                                     cmap='binary_r', origin='lower')
                boundary.plot(ax=axes[1], color='red', linewidth=1)
                axes[1].set_title('Data Coverage\n(White = Valid Data)')
                axes[1].set_xlabel('Longitude')
                plt.colorbar(im2, ax=axes[1])

                # Plot normalized data
                vmin = -1 if indicator_type in ['v', 'both'] else 0
                vmax = 1 if indicator_type in ['r', 'both'] else 0

                # Choose colormap based on indicator type
                if indicator_type == 'v':
                    cmap = 'Reds_r'  # Reversed so darker red means more vulnerable
                elif indicator_type == 'r':
                    cmap = 'Greens'  # Darker green means more resilient
                    if prerequisite_flag:
                        cmap = 'Blues'
                else:
                    cmap = 'RdYlGn'  # Red for vulnerability, green for resilience

                im3 = axes[2].imshow(normalized_data, extent=extent,
                                     cmap=cmap, vmin=vmin, vmax=vmax,
                                     origin='lower')
                boundary.plot(ax=axes[2], color='red', linewidth=1)
                title = 'Normalized Data '
                if indicator_type == 'v':
                    title += '[-1, 0] (Vulnerability)'
                elif indicator_type == 'r':
                    if prerequisite_flag:
                        title += '[0, 1] (Prerequisite)'
                    else:
                        title += '[0, 1] (Resilience)'
                else:
                    title += '[-1, 1] (Vulnerability-Resilience)'
                axes[2].set_title(title)
                axes[2].set_xlabel('Longitude')
                plt.colorbar(im3, ax=axes[2])

                plt.tight_layout()
                plt.show()

            return normalized_data, data_range

    except Exception as e:
        print(f"Error processing TIFF file {tiff_path}: {str(e)}")
        return None, None


# ---------------------- RESPONSE FUNCTIONS -----------------------------------------------------------------------
# **************************************************************************************************************

def linear_response(x: np.ndarray, m: float = 1.0) -> np.ndarray:
    """Linear response function: f(x) = m*x"""
    return m * x


def logarithmic_response(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Logarithmic response function: f(x) = ln(1 + αx)"""
    return np.log1p(alpha * x)


def exponential_response(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Exponential response function: f(x) = exp(βx) - 1"""
    return np.exp(beta * x) - 1


def power_law_response(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power law response function: f(x) = x^γ"""
    return x ** gamma


def sigmoid_response(x: np.ndarray, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """Sigmoid response function: f(x) = 1/(1 + exp(-k(x - x₀)))"""
    return 1 / (1 + np.exp(-k * (x - x0)))


# ---------------------- DEPENDENCY FUNCTIONS ---------------------------------------------------------------------
# **************************************************************************************************************

def apply_threshold_dependency(prerequisite_data: np.ndarray, base_data: np.ndarray,
                               threshold: float, base_level: float, reverse: bool = False) -> np.ndarray:
    """
    Apply threshold dependency where base data effectiveness depends on prerequisite data reaching a threshold.

    Parameters:
        prerequisite_data (np.ndarray): Already mapped values for the prerequisite indicator
        base_data (np.ndarray): Already mapped values for the dependent indicator
        threshold (float): Level of prerequisite indicator required for threshold effect
        base_level (float): Base effectiveness level when threshold condition is met/not met
        reverse (bool): If True, base_level applies when prerequisite exceeds threshold
                       If False, base_level applies when prerequisite is below threshold

    Returns:
        np.ndarray: Modified response based on threshold dependency
    """
    mask = prerequisite_data >= threshold
    if reverse:
        mask = ~mask
    return np.where(mask, base_data, base_data * base_level)


def apply_amplifying_dependency(prerequisite_data: np.ndarray, base_data: np.ndarray,
                                coupling_strength: float, coupling_ranges: Optional[tuple] = None) -> np.ndarray:
    """
    Apply amplifying/dampening dependency where prerequisite data modifies base data effectiveness.

    Parameters:
        prerequisite_data (np.ndarray): Already mapped values for the modifying indicator
        base_data (np.ndarray): Already mapped values for the base indicator
        coupling_strength (float): Base coupling coefficient or threshold(s) for ranged coupling
        coupling_ranges (tuple, optional): Either (c_low, c_high) for single threshold or
                                          (c_low, c_mid, c_high) for two thresholds

    Returns:
        np.ndarray: Modified response based on amplifying dependency
    """
    if coupling_ranges is None:
        return base_data * (1 + prerequisite_data * coupling_strength)

    if len(coupling_ranges) == 2:
        # Single threshold case
        c_low, c_high = coupling_ranges
        mask_low = prerequisite_data < coupling_strength
        return np.where(mask_low,
                        base_data * (1 + prerequisite_data * c_low),
                        base_data * (1 + prerequisite_data * c_high))
    else:
        # Two thresholds case
        c_low, c_mid, c_high = coupling_ranges
        t1, t2 = coupling_strength  # coupling_strength should be tuple of thresholds

        mask_low = prerequisite_data < t1
        mask_mid = (prerequisite_data >= t1) & (prerequisite_data < t2)

        result = np.zeros_like(base_data)
        result[mask_low] = base_data[mask_low] * (1 + prerequisite_data[mask_low] * c_low)
        result[mask_mid] = base_data[mask_mid] * (1 + prerequisite_data[mask_mid] * c_mid)
        result[~(mask_low | mask_mid)] = base_data[~(mask_low | mask_mid)] * (
                    1 + prerequisite_data[~(mask_low | mask_mid)] * c_high)

        return result


def test_threshold_dependency():
    """Test function to demonstrate threshold dependency behavior."""
    # Create sample grid data
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Create sample data for both indicators
    travel_time_data = (X + 1) / 2  # Normalize to 0-1 range
    infrastructure_data = np.ones_like(X) * 0.8  # Constant infrastructure value

    # Apply threshold dependency
    result = apply_threshold_dependency(
        travel_time_data,
        infrastructure_data,
        threshold=0.3,
        base_level=0.4,
        reverse=True
    )

    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    im1 = ax1.imshow(travel_time_data, cmap='viridis')
    ax1.set_title('Travel Time Data')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(infrastructure_data, cmap='viridis')
    ax2.set_title('Infrastructure Data')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(result, cmap='viridis')
    ax3.set_title('Result (with threshold)')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()


# ---------------------- CONFIGURATION CLASSES AND ENUMS ---------------------------------------------------------
# **************************************************************************************************************

class ParameterType(Enum):
    """Physical parameter types for FEA material properties."""
    THICKNESS = 'thickness'
    YOUNGS_MODULUS = 'youngs_modulus'
    POISSON_RATIO = 'poisson_ratio'


class ResponseType(Enum):
    """Mathematical response function types for indicator mapping."""
    LINEAR = 'linear'
    LOGARITHMIC = 'logarithmic'
    EXPONENTIAL = 'exponential'
    POWER_LAW = 'power_law'
    SIGMOID = 'sigmoid'


class RangeType(Enum):
    """Data range types for indicator normalization."""
    VULNERABILITY = 'vulnerability'  # For mapping negative range [-1,0]
    RESILIENCE = 'resilience'  # For mapping positive range [0,1]
    PREREQUISITE = 'prerequisite'


@dataclass
class ResponseConfig:
    """Configuration for response function mapping."""
    function_type: ResponseType
    parameters: dict
    range_type: RangeType


@dataclass
class PrerequisiteConfig:
    """Configuration for dependency relationships between indicators."""
    indicator_name: str
    response_config: ResponseConfig
    threshold: Optional[float] = None
    base_level: Optional[float] = 0.1  # Default base level
    reverse: bool = False  # Whether to reverse threshold effect
    coupling_strength: Optional[float] = None  # For amplifying dependency
    coupling_ranges: Optional[tuple] = None  # For ranged amplifying dependency


@dataclass
class ParameterMapping:
    """Mapping configuration from indicator to physical parameter."""
    parameter_type: ParameterType
    response_config: ResponseConfig
    weight: float
    prerequisite_config: Optional[PrerequisiteConfig] = None


# ---------------------- INDICATOR CLASS --------------------------------------------------------------------------
# **************************************************************************************************************

class Indicator:
    """
    Represents a social or physical indicator that can be mapped to FEA material parameters.

    This class handles the loading, normalization, and mapping of indicator data
    from TIFF files to material properties used in finite element analysis.
    """

    def __init__(self, name: str, tiff_path: str, indicator_type: str, mid_point: Optional[float] = None,
                 resampling_method: str = 'average', fill_nan: float = 0.0, flip_data: bool = False):
        """
        Initialize an indicator with its configuration.

        Parameters:
            name (str): Unique identifier for the indicator
            tiff_path (str): Path to the TIFF file containing the data
            indicator_type (str): 'v' for vulnerability, 'r' for resilience, 'both' for both
            mid_point (float, optional): Value used as the midpoint for normalization when indicator_type is 'both'
            resampling_method (str): Method for resampling the TIFF data
            fill_nan (float): Value to use for areas without data
            flip_data (bool): Whether to flip the sign of the data before normalization
        """
        self.name = name
        self.tiff_path = tiff_path
        self.indicator_type = indicator_type
        self.mid_point = mid_point
        self.resampling_method = resampling_method
        self.fill_nan = fill_nan
        self.flip_data = flip_data

        self.normalized_data = None
        self.data_range = None
        self.parameter_mappings: Dict[ParameterType, Dict[RangeType, ParameterMapping]] = {}

    def add_parameter_mapping(self, parameter_type, response_config: ResponseConfig,
                              weight: float, prerequisite_config: Optional[PrerequisiteConfig] = None):
        """
        Add a mapping from this indicator to a physical parameter.

        Parameters:
            parameter_type (ParameterType or str): The physical parameter this indicator contributes to
            response_config (ResponseConfig): Configuration for the response function
            weight (float): Weight of this indicator's contribution to the parameter
            prerequisite_config (PrerequisiteConfig, optional): Configuration for dependencies on other indicators
        """
        if isinstance(parameter_type, str):
            parameter_type = ParameterType(parameter_type)

        # Validate range type against indicator type
        if self.indicator_type == 'v' and response_config.range_type != RangeType.VULNERABILITY:
            raise ValueError(f"Vulnerability indicator can only have VULNERABILITY range type")
        if self.indicator_type == 'r' and response_config.range_type != RangeType.RESILIENCE:
            raise ValueError(f"Resilience indicator can only have RESILIENCE range type")
        if self.indicator_type == 'p' and response_config.range_type != RangeType.PREREQUISITE:
            raise ValueError(f"Prerequisite indicator can only have PREREQUISITE range type")

        mapping = ParameterMapping(
            parameter_type=parameter_type,
            response_config=response_config,
            weight=weight,
            prerequisite_config=prerequisite_config
        )

        if parameter_type not in self.parameter_mappings:
            self.parameter_mappings[parameter_type] = {}
        self.parameter_mappings[parameter_type][response_config.range_type] = mapping


# ---------------------- INDICATOR MANAGER CLASS -----------------------------------------------------------------
# **************************************************************************************************************

class IndicatorManager:
    """
    Manages multiple indicators and their mapping to FEA material parameters.

    This class coordinates the loading of indicator data, application of response functions,
    handling of dependencies, and final computation of material properties for FEA analysis.
    """

    def __init__(self, mesh, base_thickness: float, base_youngs_modulus: float, poisson_ratio_k: float = 2.5):
        """
        Initialize the indicator manager.

        Parameters:
            mesh (FEAMesh2D): Mesh object containing geometry and projection information
            base_thickness (float): Base value for thickness parameter
            base_youngs_modulus (float): Base value for Young's modulus parameter
            poisson_ratio_k (float): Sensitivity parameter k for Poisson's ratio sigmoid mapping
        """
        self.mesh = mesh
        self.indicators: Dict[str, Indicator] = {}
        self.base_thickness = base_thickness
        self.base_youngs_modulus = base_youngs_modulus
        self.poisson_ratio_k = poisson_ratio_k

        self.response_functions = {
            ResponseType.LINEAR: linear_response,
            ResponseType.LOGARITHMIC: logarithmic_response,
            ResponseType.EXPONENTIAL: exponential_response,
            ResponseType.POWER_LAW: power_law_response,
            ResponseType.SIGMOID: sigmoid_response
        }

    def add_indicator(self, indicator: Indicator, validate: bool = False):
        """
        Add an indicator to the manager.

        Parameters:
            indicator (Indicator): The indicator to add
            validate (bool): Whether to validate weights immediately
        """
        self.indicators[indicator.name] = indicator

        # Validate mappings for 'both' type indicators
        if indicator.indicator_type == 'both':
            for param_type in indicator.parameter_mappings:
                mappings = indicator.parameter_mappings[param_type]
                missing_ranges = []

                if RangeType.VULNERABILITY not in mappings:
                    missing_ranges.append('vulnerability')
                if RangeType.RESILIENCE not in mappings:
                    missing_ranges.append('resilience')

                if missing_ranges:
                    raise ValueError(f"Indicator '{indicator.name}' missing {', '.join(missing_ranges)} mappings")

        # Only validate weights if requested
        if validate:
            weight_sums = self.validate_weights()
            for param_type, weight_sum in weight_sums.items():
                print(f"Current weight sum for {param_type.value}: {weight_sum:.3f}")

    def load_indicator_data(self, indicator_name: str):
        """Load and normalize data for a specific indicator."""
        indicator = self.indicators[indicator_name]

        normalized_data, data_range = import_and_normalize_tiff(
            mesh=self.mesh,
            tiff_path=indicator.tiff_path,
            indicator_type=indicator.indicator_type,
            mid_point=indicator.mid_point,
            resampling_method=indicator.resampling_method,
            fill_nan=indicator.fill_nan,
            flip_data=indicator.flip_data
        )
        indicator.normalized_data = normalized_data
        indicator.data_range = data_range

    def apply_response_function(self, data: np.ndarray, response_config: ResponseConfig):
        """Apply response function to data with proper range handling."""
        result = np.zeros_like(data)

        if response_config.range_type == RangeType.VULNERABILITY:
            mask = data < 0
            # Take absolute value for response function and reapply negative sign
            result[mask] = -1 * self.response_functions[response_config.function_type](
                np.abs(data[mask]), **response_config.parameters
            )
        else:  # RangeType.RESILIENCE or RangeType.PREREQUISITE
            mask = data >= 0
            result[mask] = self.response_functions[response_config.function_type](
                data[mask], **response_config.parameters
            )

        return result

    def apply_dependency(self, base_data: np.ndarray, config: PrerequisiteConfig) -> np.ndarray:
        """Apply dependency modification using PrerequisiteConfig."""
        prerequisite_data = self.indicators[config.indicator_name].normalized_data

        # Apply response function to prerequisite data first
        prerequisite_processed = self.apply_response_function(
            prerequisite_data,
            config.response_config
        )

        if config.threshold is not None:
            return apply_threshold_dependency(
                prerequisite_processed,
                base_data,
                config.threshold,
                config.base_level,
                config.reverse
            )
        else:
            return apply_amplifying_dependency(
                prerequisite_processed,
                base_data,
                config.coupling_strength,
                config.coupling_ranges
            )

    def validate_weights(self):
        """
        Validate that weights sum to 1.0 for each parameter type that has mappings.

        Returns:
            dict: Dictionary with parameter types as keys and their weight sums as values

        Raises:
            ValueError: If weights for any parameter type do not sum to 1.0
        """
        weight_sums = {}
        for param_type in ParameterType:
            weight_sum = 0.0
            has_mappings = False

            for indicator in self.indicators.values():
                if param_type in indicator.parameter_mappings:
                    has_mappings = True
                    mappings = indicator.parameter_mappings[param_type]
                    for mapping in mappings.values():
                        weight_sum += mapping.weight

            if has_mappings:
                weight_sums[param_type] = weight_sum
                if not np.isclose(weight_sum, 1.0, rtol=1e-5):
                    raise ValueError(
                        f"Weights for {param_type.value} sum to {weight_sum:.3f}, not 1.0"
                    )

        return weight_sums

    def validate_both_ranges(self):
        """Validate that indicators of type 'both' have mappings for both ranges."""
        for indicator in self.indicators.values():
            if indicator.indicator_type == 'both':
                for param_type in indicator.parameter_mappings:
                    mappings = indicator.parameter_mappings[param_type]
                    if RangeType.VULNERABILITY not in mappings or RangeType.RESILIENCE not in mappings:
                        raise ValueError(
                            f"Indicator {indicator.name} of type 'both' must have mappings "
                            f"for both vulnerability and resilience ranges for parameter {param_type.value}"
                        )

    def compute_parameter(self, parameter_type: ParameterType):
        """
        Compute final parameter values from all contributing indicators.

        Parameters:
            parameter_type (ParameterType): The physical parameter to compute

        Returns:
            np.ndarray: Combined parameter values

        Raises:
            ValueError: If no indicators are added, data is not loaded,
                       mappings are incomplete, or weights don't sum to 1.0
        """
        # Check if there are any indicators
        if not self.indicators:
            raise ValueError("No indicators added to manager")

        # Check if all indicators have data loaded and proper mappings
        for indicator in self.indicators.values():
            if indicator.normalized_data is None:
                raise ValueError(f"Data not loaded for indicator {indicator.name}. "
                                 f"Call load_indicator_data() first.")

            # Verify complete mappings for 'both' type indicators
            if (indicator.indicator_type == 'both' and
                    parameter_type in indicator.parameter_mappings):
                mappings = indicator.parameter_mappings[parameter_type]
                missing_ranges = []

                if RangeType.VULNERABILITY not in mappings:
                    missing_ranges.append('vulnerability')
                if RangeType.RESILIENCE not in mappings:
                    missing_ranges.append('resilience')

                if missing_ranges:
                    raise ValueError(
                        f"Indicator '{indicator.name}' is of type 'both' but missing "
                        f"{', '.join(missing_ranges)} mapping(s) for parameter "
                        f"'{parameter_type.value}'. Both ranges must be mapped for "
                        f"indicators of type 'both'."
                    )

        # Validate weights before computation
        self.validate_weights()

        weighted_sum = np.zeros_like(next(iter(self.indicators.values())).normalized_data)

        for indicator in self.indicators.values():
            if parameter_type in indicator.parameter_mappings:
                mappings = indicator.parameter_mappings[parameter_type]
                for mapping in mappings.values():
                    # Apply response function
                    mapped_data = self.apply_response_function(
                        indicator.normalized_data,
                        mapping.response_config
                    )

                    # Apply dependency if specified
                    if mapping.prerequisite_config:
                        mapped_data = self.apply_dependency(
                            mapped_data,
                            mapping.prerequisite_config
                        )

                    weighted_sum += mapping.weight * mapped_data

        # Apply final parameter combination
        if parameter_type == ParameterType.THICKNESS:
            return self.base_thickness * np.exp(weighted_sum)
        elif parameter_type == ParameterType.YOUNGS_MODULUS:
            return self.base_youngs_modulus * np.exp(weighted_sum)
        else:  # Poisson's ratio
            return 0.5 / (1 + np.exp(self.poisson_ratio_k * weighted_sum))

    def visualize_parameters(self, parameter_values: Dict[ParameterType, np.ndarray], single_row: bool = True):
        """
        Create visualizations for the computed parameters.

        Parameters:
            parameter_values (Dict[ParameterType, np.ndarray]): Dictionary containing computed values for each parameter type
            single_row (bool): If True, displays all parameters in one row; if False, displays each parameter separately
        """
        # Define colormaps and titles for each parameter
        param_configs = {
            ParameterType.YOUNGS_MODULUS: {
                'title': "Young's Modulus Distribution",
                'cmap': 'Blues',  # Blue colormap
                'label': "Young's modulus (Pa)"
            },
            ParameterType.POISSON_RATIO: {
                'title': "Poisson's Ratio Distribution",
                'cmap': 'Purples',  # Purple colormap
                'label': "Poisson's ratio"
            },
            ParameterType.THICKNESS: {
                'title': "Thickness Distribution",
                'cmap': 'Oranges',  # Orange colormap
                'label': "Thickness"
            }
        }

        if single_row:
            # Setup single row figure
            fig, axes = plt.subplots(1, len(parameter_values), figsize=(20, 6))
            if len(parameter_values) == 1:
                axes = [axes]

            for ax, (param_type, values) in zip(axes, parameter_values.items()):
                self._plot_parameter(ax, values, param_configs[param_type])

            plt.tight_layout()
            plt.show()

        else:
            # Create separate figure for each parameter
            for param_type, values in parameter_values.items():
                fig, ax = plt.subplots(figsize=(10, 8))
                self._plot_parameter(ax, values, param_configs[param_type])
                plt.tight_layout()
                plt.show()

    def _plot_parameter(self, ax, values, config):
        """
        Plot a single parameter visualization.

        Parameters:
            ax (matplotlib.axes.Axes): The axes to plot on
            values (np.ndarray): The values to plot
            config (dict): Configuration dictionary with plot settings
        """
        # Create the main visualization
        extent = [self.mesh.bounds[0], self.mesh.bounds[2],
                  self.mesh.bounds[1], self.mesh.bounds[3]]
        im = ax.imshow(values, extent=extent,
                       cmap=config['cmap'], origin='lower')

        # Add original boundary
        boundary = self.mesh.gdf_original.boundary
        if self.mesh.region_crs != self.mesh.gdf_original.crs:
            boundary = boundary.to_crs(self.mesh.region_crs)
        boundary.plot(ax=ax, color='red', linewidth=1, label='Original Region')

        # Customize the plot
        ax.set_title(config['title'], fontsize=12, pad=10)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)

        # Set consistent tick parameters
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config['label'], fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # Add legend with small size
        ax.legend(loc='upper right', fontsize=8)


# ---------------------- LAYER PROPERTIES CLASS ------------------------------------------------------------------
# **************************************************************************************************************

class LayerProperties:
    """
    Class to store and combine material properties for plate bending analysis.

    This class handles three main properties:
    - Young's modulus (E)
    - Poisson's ratio (nu)
    - Thickness (h)

    Each property can vary spatially across the plate, stored as 2D numpy arrays.
    """

    def __init__(self):
        self.layers = {}
        self.E_layers = {}
        self.nu_layers = {}
        self.thickness_layers = {}

    def _add_layer(self, name: str, E: np.ndarray, nu: np.ndarray, thickness: np.ndarray):
        """
        Add a new layer of material properties with validation.

        Parameters:
            name (str): Identifier for the layer
            E (np.ndarray): Young's modulus field (2D array)
            nu (np.ndarray): Poisson's ratio field (2D array)
            thickness (np.ndarray): Thickness field (2D array)
        """
        # Validate material properties
        if np.any(E <= 0):
            raise ValueError("Young's modulus must be positive!")
        if np.any((nu < -1.0) | (nu > 0.5)):
            raise ValueError("Poisson's ratio must be between -1.0 and 0.5 for stable plates!")
        if np.any(thickness <= 0):
            raise ValueError("Thickness must be positive!")

        # Store properties
        self.E_layers[name] = E
        self.nu_layers[name] = nu
        self.thickness_layers[name] = thickness

    def _verify_plate_stability(self, E: np.ndarray, nu: np.ndarray, thickness: np.ndarray) -> bool:
        """
        Verify if the material properties satisfy plate stability conditions.

        This function checks both material property ranges and plate theory
        applicability based on thickness-to-span ratios.
        """
        # Check basic ranges
        if np.any(E <= 0):
            return False
        if np.any((nu < -1.0) | (nu > 0.5)):
            return False
        if np.any(thickness <= 0):
            return False

        # Check plate stability criterion
        # For plates, typically -1 < ν < 0.5
        # E should be positive
        stability_criterion = E > 0
        return np.all(stability_criterion)

    def _combine_properties(self, operation: str = 'weighted_sum', weights: dict = None,
                            custom_function_E: callable = None,
                            custom_function_nu: callable = None,
                            custom_function_thickness: callable = None,
                            ) -> tuple:
        """
        Combine stored material properties using specified operation.

        Parameters:
            operation (str): Combination method ('weighted_sum', 'multiply', 'maximum', 'minimum', 'average', or 'custom')
            weights (dict, optional): Layer weights for weighted sum operation
            custom_function_* (callable, optional): Custom combining functions for each property

        Returns:
            tuple: (E_combined, nu_combined, thickness_combined) Combined property fields
        """
        if not self.E_layers:
            raise ValueError("No layers added yet")

        # Apply combination logic based on operation type
        if operation == 'weighted_sum':
            if not weights:
                raise ValueError("Weights required for weighted sum")

            E_combined = np.zeros_like(list(self.E_layers.values())[0])
            nu_combined = np.zeros_like(list(self.nu_layers.values())[0])
            thickness_combined = np.zeros_like(list(self.thickness_layers.values())[0])

            for name in self.E_layers.keys():
                E_combined += self.E_layers[name] * weights[name]
                nu_combined += self.nu_layers[name] * weights[name]
                thickness_combined += self.thickness_layers[name] * weights[name]

        elif operation == 'multiply':
            E_combined = np.ones_like(list(self.E_layers.values())[0])
            nu_combined = np.ones_like(list(self.nu_layers.values())[0])
            thickness_combined = np.ones_like(list(self.thickness_layers.values())[0])

            for name in self.E_layers.keys():
                E_combined *= self.E_layers[name]
                nu_combined *= self.nu_layers[name]
                thickness_combined *= self.thickness_layers[name]

        elif operation == 'maximum':
            E_combined = np.maximum.reduce(list(self.E_layers.values()))
            nu_combined = np.maximum.reduce(list(self.nu_layers.values()))
            thickness_combined = np.maximum.reduce(list(self.thickness_layers.values()))

        elif operation == 'minimum':
            E_combined = np.minimum.reduce(list(self.E_layers.values()))
            nu_combined = np.minimum.reduce(list(self.nu_layers.values()))
            thickness_combined = np.minimum.reduce(list(self.thickness_layers.values()))

        elif operation == 'average':
            E_combined = np.mean(list(self.E_layers.values()), axis=0)
            nu_combined = np.mean(list(self.nu_layers.values()), axis=0)
            thickness_combined = np.mean(list(self.thickness_layers.values()), axis=0)

        elif operation == 'custom':
            if not custom_function_E or not custom_function_nu:
                raise ValueError("Custom functions required for custom operation")
            E_combined = custom_function_E(self.E_layers)
            nu_combined = custom_function_nu(self.nu_layers)
            thickness_combined = custom_function_thickness(self.thickness_layers)

        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Verify plate stability of combined properties
        if not self._verify_plate_stability(E_combined, nu_combined, thickness_combined):
            warnings.warn("Combined properties may not satisfy plate stability conditions")

        return E_combined, nu_combined, thickness_combined


# ---------------------- UTILITY FUNCTIONS ------------------------------------------------------------------------
# **************************************************************************************************************

def process_multiple_layers(mesh, layer_configs: list) -> LayerProperties:
    """
    Process multiple TIFF layers for plate bending analysis using existing import_tiff_properties.

    Parameters:
        mesh (FEAMesh2D): The mesh object for plate bending analysis
        layer_configs (list): List of configurations for each layer containing name, tiff_path,
                             property_mapping_function, resampling_method, fill_nan, and scale_factor

    Returns:
        LayerProperties: Object containing all processed layers with validated plate properties
    """
    layers = LayerProperties()

    for config in layer_configs:
        # Store current property arrays
        E_original = mesh.E.copy()
        nu_original = mesh.nu.copy()
        thickness_original = mesh.thickness.copy()

        # Import this layer's properties
        mesh._import_tiff_properties(
            tiff_path=config['tiff_path'],
            property_mapping_function=config['property_mapping_function'],
            resampling_method=config.get('resampling_method', 'bilinear'),
            validate=True,
            visualize=False,
            fill_nan=config.get('fill_nan', 0.0),
            scale_factor=config.get('scale_factor', 1.0)
        )

        # Store the layer's properties
        layers._add_layer(
            config['name'],
            mesh.E.copy(),
            mesh.nu.copy(),
            mesh.thickness.copy()
        )

        # Restore original properties
        mesh.E = E_original
        mesh.nu = nu_original
        mesh.thickness = thickness_original

    return layers


def generate_test_layers(mesh, pattern_config: dict) -> LayerProperties:
    """
    Generate test patterns for material properties with flexible value specification
    and support for social structure modeling.

    Parameters:
        mesh (FEAMesh2D): The mesh object containing geometry information
        pattern_config (dict): Configuration specifying the pattern type and parameters

    Returns:
        LayerProperties: Object containing generated test pattern properties
    """
    layers = LayerProperties()
    ny, nx = mesh.ny, mesh.nx
    pattern_type = pattern_config.get('pattern_type', 'uniform')

    def map_to_range(normalized_values, value_range):
        """Map normalized values (0-1) to specified range"""
        return value_range[0] + normalized_values * (value_range[1] - value_range[0])

    def validate_properties(E, nu, thickness):
        """Validate material property values"""
        if E <= 0:
            raise ValueError("Young's modulus must be positive.")
        if nu < 0 or nu > 0.5:
            raise ValueError("Poisson's ratio must be between 0 and 0.5.")
        if thickness <= 0:
            raise ValueError("Thickness must be positive.")

    def validate_ranges(E_range, nu_range, thickness_range):
        """Validate material property ranges"""
        if E_range[0] <= 0 or E_range[1] <= E_range[0]:
            raise ValueError("Invalid E range. Must be positive with min < max.")
        if nu_range[0] < 0 or nu_range[1] > 0.5 or nu_range[1] <= nu_range[0]:
            raise ValueError("Invalid nu range. Must be between 0 and 0.5 with min < max.")
        if thickness_range[0] <= 0 or thickness_range[1] <= thickness_range[0]:
            raise ValueError("Invalid thickness range. Must be positive with min < max.")

    if pattern_type == 'uniform':
        # Create uniform property distribution
        E_value = pattern_config.get('E', 5e9)
        nu_value = pattern_config.get('nu', 0.3)
        thickness_value = pattern_config.get('thickness', 2_000)

        validate_properties(E_value, nu_value, thickness_value)

        E = np.ones((ny, nx)) * E_value
        nu = np.ones((ny, nx)) * nu_value
        thickness = np.ones((ny, nx)) * thickness_value

    elif pattern_type == 'split':
        # Create sharp division pattern (north-south or east-west)
        split_direction = pattern_config.get('split_direction', 'ns')
        split_position = pattern_config.get('split_position', 0.5)

        # Initialize arrays
        E = np.zeros((ny, nx))
        nu = np.zeros((ny, nx))
        thickness = np.zeros((ny, nx))

        if split_direction == 'ns':  # North-South split
            split_idx = int(ny * split_position)
            # Apply properties to each half
            for prop, arr in [('thickness', thickness), ('E', E), ('nu', nu)]:
                arr[:split_idx, :] = pattern_config.get('south_properties', {}).get(prop, arr[:split_idx, :])
                arr[split_idx:, :] = pattern_config.get('north_properties', {}).get(prop, arr[split_idx:, :])
        else:  # East-West split
            split_idx = int(nx * split_position)
            # Apply properties to each half
            for prop, arr in [('thickness', thickness), ('E', E), ('nu', nu)]:
                arr[:, :split_idx] = pattern_config.get('west_properties', {}).get(prop, arr[:, :split_idx])
                arr[:, split_idx:] = pattern_config.get('east_properties', {}).get(prop, arr[:, split_idx:])

    elif pattern_type == 'center_zone':
        # Create coordinate grids matching the property array dimensions
        # Use geographic coordinates for Nigeria
        lons = np.linspace(2, 16, nx)  # Longitude range
        lats = np.linspace(3, 15, ny)  # Latitude range
        LON, LAT = np.meshgrid(lons, lats)

        # Define center point
        center_lon = 9.00
        center_lat = 9.00

        # Get radius in degrees
        radius = pattern_config.get('center_radius', 1.0)

        # Create mask using geographic distance
        # Using approximate spherical distance
        mask = ((LON - center_lon) ** 2 + (LAT - center_lat) ** 2) <= radius ** 2

        # Set properties with mask
        outer_props = pattern_config.get('outer_properties', {})
        center_props = pattern_config.get('center_properties', {})

        E = np.full((ny, nx), outer_props.get('E', 5e9))
        nu = np.full((ny, nx), outer_props.get('nu', 0.3))
        thickness = np.full((ny, nx), outer_props.get('thickness', 2000))

        # Apply center properties using mask
        E[mask] = center_props.get('E', 1e10)
        nu[mask] = center_props.get('nu', 0.4)
        thickness[mask] = center_props.get('thickness', 5000)

    else:
        # Handle pattern types that require ranges
        E_range = pattern_config.get('E_range', (1e9, 5e9))
        nu_range = pattern_config.get('nu_range', (0.2, 0.4))
        thickness_range = pattern_config.get('thickness_range', (2_000, 3_000))

        validate_ranges(E_range, nu_range, thickness_range)

        if pattern_type == 'gradient_x':
            x = np.linspace(0, 1, nx)
            gradient = np.tile(x, (ny, 1))
            E = map_to_range(gradient, E_range)
            nu = map_to_range(gradient, nu_range)
            thickness = map_to_range(gradient, thickness_range)

        elif pattern_type == 'gradient_y':
            y = np.linspace(0, 1, ny)[:, np.newaxis]
            gradient = np.tile(y, (1, nx))
            E = map_to_range(gradient, E_range)
            nu = map_to_range(gradient, nu_range)
            thickness = map_to_range(gradient, thickness_range)

        elif pattern_type == 'radial':
            x = np.linspace(-1, 1, nx)
            y = np.linspace(-1, 1, ny)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X ** 2 + Y ** 2)
            R = R / np.max(R)
            E = map_to_range(1 - R, E_range)
            nu = map_to_range(1 - R, nu_range)
            thickness = map_to_range(1 - R, thickness_range)

        elif pattern_type == 'checkerboard':
            square_size = pattern_config.get('checkerboard_size', 5)
            x = np.arange(nx) // square_size
            y = np.arange(ny) // square_size
            X, Y = np.meshgrid(x, y)
            checker = (X + Y) % 2
            E = map_to_range(checker, E_range)
            nu = map_to_range(checker, nu_range)
            thickness = map_to_range(checker, thickness_range)

        elif pattern_type == 'bands':
            num_bands = pattern_config.get('num_bands', 5)
            orientation = pattern_config.get('orientation', 'horizontal')

            if orientation == 'horizontal':
                y = np.linspace(0, num_bands, ny)
                pattern = 0.5 * (1 + np.cos(2 * np.pi * y))
                pattern = np.tile(pattern[:, np.newaxis], (1, nx))
            else:
                x = np.linspace(0, num_bands, nx)
                pattern = 0.5 * (1 + np.cos(2 * np.pi * x))
                pattern = np.tile(pattern, (ny, 1))

            E = map_to_range(pattern, E_range)
            nu = map_to_range(pattern, nu_range)
            thickness = map_to_range(pattern, thickness_range)

        elif pattern_type == 'random':
            correlation_length = pattern_config.get('correlation_length', 5)
            from scipy.ndimage import gaussian_filter

            E_random = np.random.randn(ny, nx)
            nu_random = np.random.randn(ny, nx)
            thickness_random = np.random.randn(ny, nx)

            E_smooth = gaussian_filter(E_random, sigma=correlation_length)
            nu_smooth = gaussian_filter(nu_random, sigma=correlation_length)
            thickness_smooth = gaussian_filter(thickness_random, sigma=correlation_length)

            E_norm = (E_smooth - E_smooth.min()) / (E_smooth.max() - E_smooth.min())
            nu_norm = (nu_smooth - nu_smooth.min()) / (nu_smooth.max() - nu_smooth.min())
            thickness_norm = (thickness_smooth - thickness_smooth.min()) / (
                        thickness_smooth.max() - thickness_smooth.min())

            E = map_to_range(E_norm, E_range)
            nu = map_to_range(nu_norm, nu_range)
            thickness = map_to_range(thickness_norm, thickness_range)

    # Add to layers
    layers._add_layer('test_pattern', E, nu, thickness)
    return layers


def verify_mesh_creation(mesh):
    """Verify that mesh has been created correctly with expected dimensions."""
    expected_nodes = (mesh.nx + 1) * (mesh.ny + 1)
    expected_dofs = expected_nodes * 3

    print("Mesh Verification:")
    print(f"Elements: {mesh.nx} x {mesh.ny}")
    print(f"Expected nodes: {expected_nodes}, Actual: {mesh.nnodes}")
    print(f"Expected DOFs: {expected_dofs}, Actual: {mesh.ndof}")

    if mesh.nnodes != expected_nodes or mesh.ndof != expected_dofs:
        print("ERROR: Mesh node/DOF count mismatch!")