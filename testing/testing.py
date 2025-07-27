import unittest
from fea_mesh_2d import FEAMesh2D
from conflict_translation import apply_multiple_forces
from scipy.linalg import eigvalsh
import numpy as np
import matplotlib.pyplot as plt


# ---------------------- MAIN TEST CLASS --------------------------------------------------------------------------
# **************************************************************************************************************

class FEATests(unittest.TestCase):
    """
    Test suite for plate bending FEA implementation.

    This class provides comprehensive testing of the finite element implementation,
    covering mathematical verification and physical behavior validation.
    """

    def __init__(self, methodName='runTest'):
        """
        Initialize test configuration with physically meaningful parameters.

        Parameters:
            methodName (str): Name of the test method to run
        """
        super().__init__(methodName)

        # Physical parameters (steel plate example)
        self.width = 1.0  # meters
        self.height = 1.0  # meters
        self.thickness = 0.01  # meters
        self.E = 200e9  # Young's modulus (Pa)
        self.nu = 0.3  # Poisson's ratio

        # Mesh parameters
        self.nx = 8
        self.ny = 8

        # Initialize mesh
        self.mesh = FEAMesh2D(self.width, self.height, self.nx, self.ny)
        self.mesh.thickness.fill(self.thickness)
        self.mesh.E.fill(self.E)
        self.mesh.nu.fill(self.nu)

        # Calculate important physical parameters
        self.D = self.E * self.thickness ** 3 / (12 * (1 - self.nu ** 2))  # Flexural rigidity
        self.characteristic_length = min(self.width, self.height)
        self.characteristic_force = self.D * self.thickness / self.characteristic_length ** 2

        # Test results tracking
        self.test_results = {}

    def run_all_tests(self):
        """
        Execute all test methods and provide comprehensive summary.

        Returns:
            bool: True if all testing pass, False otherwise
        """
        test_methods = [
            (self.test_element_mathematics, "Element Mathematics"),
            (self.test_physical_behavior, "Physical Behavior"),
            (self.test_mesh_connectivity, "Mesh Connectivity"),
            (self.test_bending_shear, "Bending-Shear Coupling"),
            (self.test_boundary_conditions, "Boundary Conditions")
        ]

        total_tests = len(test_methods)
        passed_tests = 0

        print("\nRunning FEA Tests")
        print("=" * 50)

        for test_func, test_name in test_methods:
            print(f"\nExecuting {test_name} Test...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    passed_tests += 1
                    print(f"{test_name}: PASS")
                else:
                    print(f"{test_name}: FAIL")
            except Exception as e:
                self.test_results[test_name] = False
                print(f"{test_name}: FAIL (Error: {str(e)})")

        print("\nTest Summary")
        print("=" * 50)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        for name, result in self.test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {name}")

        return passed_tests == total_tests

    # ---------------------- MATHEMATICAL VERIFICATION TESTS ---------------------------------------------------------
    # **************************************************************************************************************

    def test_element_mathematics(self):
        """
        Verify mathematical properties of the plate bending element formulation.

        This test examines the element stiffness matrix structure and eigenvalue
        spectrum to ensure proper representation of physical deformation modes.

        Returns:
            bool: True if all mathematical checks pass
        """
        K = self.mesh._element_stiffness(self.nx // 2, self.ny // 2)
        eigenvals = eigvalsh(K)

        # Create visualization of matrix properties
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Stiffness matrix heatmap
        im1 = ax1.imshow(K, cmap='RdBu_r')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Element Stiffness Matrix')
        ax1.set_xlabel('DOF index')
        ax1.set_ylabel('DOF index')

        # Eigenvalue spectrum visualization
        sorted_eigenvals = np.sort(np.abs(eigenvals))
        ax2.semilogy(range(len(sorted_eigenvals)), sorted_eigenvals, 'bo-', label='Eigenvalues')

        # Add visual indicators for mode separation
        ax2.axvline(x=2.5, color='g', linestyle='--', label='Mode separation')
        ax2.axhspan(1e-6, 1e-2, color='y', alpha=0.2, label='Typical soft mode range')

        ax2.set_title('Eigenvalue Spectrum\n(showing deformation mode stiffness)')
        ax2.set_xlabel('Mode Index')
        ax2.set_ylabel('Eigenvalue (log scale)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Verify mathematical properties
        checks = []

        # Check matrix dimensions (12 DOFs for 4-node element with 3 DOFs per node)
        checks.append(K.shape == (12, 12))

        # Verify matrix symmetry (required for energy conservation)
        checks.append(np.max(np.abs(K - K.T)) < 1e-10)

        # Check for proper separation of deformation modes
        lowest_eigenvals = sorted_eigenvals[:3]
        higher_eigenvals = sorted_eigenvals[3:]
        eigenvalue_gap = np.min(higher_eigenvals) / np.max(lowest_eigenvals)

        # Require clear separation between soft and stiff modes
        checks.append(eigenvalue_gap > 1e3)

        # Verify positive semi-definiteness (allowing for numerical noise)
        checks.append(np.min(eigenvals) > -1e-10)

        # Print diagnostic information
        print("\nElement Stiffness Analysis:")
        print(f"Lowest eigenvalues: {lowest_eigenvals[:3]}")
        print(f"Mode separation ratio: {eigenvalue_gap:.2e}")
        print(f"Matrix conditioning: {sorted_eigenvals[-1] / sorted_eigenvals[0]:.2e}")

        return all(checks)

    def test_bending_shear(self):
        """
        Test bending-shear coupling behavior in plate elements.

        This test verifies that pure bending deformation produces
        appropriate energy distribution between bending and shear modes.

        Returns:
            bool: True if coupling behavior is within expected bounds
        """
        # Apply pure bending deformation
        u_bend = np.zeros(12)
        u_bend[1::3] = [0, 0.1, 0.1, 0]  # Linear θx variation

        K = self.mesh._element_stiffness(self.nx // 2, self.ny // 2)
        f = K @ u_bend

        # Calculate energy components
        total_energy = 0.5 * u_bend.T @ K @ u_bend
        shear_energy = 0.5 * sum(f[::3] * u_bend[::3])

        # Verify shear energy is small compared to total
        return shear_energy / total_energy < 0.1

    # ---------------------- PHYSICAL BEHAVIOR TESTS -----------------------------------------------------------------
    # **************************************************************************************************************

    def test_physical_behavior(self):
        """
        Test if solutions match expected physical behavior patterns.

        This test applies loads under different boundary conditions and verifies
        that displacement relationships follow expected physical patterns.

        Returns:
            bool: True if physical behavior is within expected ranges
        """
        force = self.characteristic_force * 10
        force_config = [{
            'coordinates': (self.width / 2, self.height / 2),
            'magnitude': force
        }]

        # Test different boundary conditions
        bc_types = ['simply_supported_all', 'clamped_all']
        results = {}

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, bc_type in enumerate(bc_types):
            fixed_dofs = self.mesh._get_boundary_nodes(bc_type)
            u = apply_multiple_forces(self.mesh, force_config, fixed_dofs)
            w = u[::3].reshape(self.ny + 1, self.nx + 1)
            results[bc_type] = np.max(np.abs(w))

            # Plot displacement field
            ax = axes[idx]
            im = ax.imshow(w, cmap='RdBu_r')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{bc_type} Displacement")

        plt.tight_layout()
        plt.show()

        # Verify physical relationships
        ratio = results['simply_supported_all'] / results['clamped_all']
        return 1.5 < ratio < 8  # Expected range for displacement ratio

    def test_boundary_conditions(self):
        """
        Test boundary condition implementation and enforcement.

        This test verifies that different boundary conditions produce
        the expected relative displacement patterns and magnitudes.

        Returns:
            bool: True if boundary condition behavior is correct
        """
        force = self.characteristic_force
        force_config = [{
            'coordinates': (self.width / 2, self.height / 2),
            'magnitude': force
        }]

        bc_types = ['simply_supported_all', 'clamped_all']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes = axes.flatten()

        results = {}
        for idx, bc_type in enumerate(bc_types):
            fixed_dofs = self.mesh._get_boundary_nodes(bc_type)
            u = apply_multiple_forces(self.mesh, force_config, fixed_dofs)
            w = u[::3].reshape(self.ny + 1, self.nx + 1)
            results[bc_type] = np.max(np.abs(w))

            # Plot displacement field
            ax = axes[idx]
            im = ax.imshow(w, cmap='RdBu_r')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{bc_type}\nMax Displacement: {results[bc_type]:.2e}")

        plt.tight_layout()
        plt.show()

        # Verify expected relationships
        checks = []

        # Simply supported should deflect more than clamped
        checks.append(results['simply_supported_all'] > results['clamped_all'])

        return all(checks)

    # ---------------------- MESH VERIFICATION TESTS -----------------------------------------------------------------
    # **************************************************************************************************************

    def test_mesh_connectivity(self):
        """
        Verify mesh connectivity and structure correctness.

        This test checks that the mesh has been constructed properly
        with correct element-node relationships and DOF mappings.

        Returns:
            bool: True if mesh connectivity is correct
        """
        # Plot mesh structure
        plt.figure(figsize=(8, 8))
        x = np.linspace(0, self.width, self.nx + 1)
        y = np.linspace(0, self.height, self.ny + 1)
        X, Y = np.meshgrid(x, y)

        plt.plot(X, Y, 'k-', alpha=0.5)
        plt.plot(X.T, Y.T, 'k-', alpha=0.5)

        # Add node numbers
        for i in range(self.ny + 1):
            for j in range(self.nx + 1):
                plt.text(x[j], y[i], f'{i * (self.nx + 1) + j}',
                         ha='center', va='center')

        plt.title('Mesh Structure with Node Numbers')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        # Verify connectivity
        checks = []

        # Element count verification
        checks.append(self.mesh.nelements == self.nx * self.ny)

        # Node count verification
        checks.append(self.mesh.nnodes == (self.nx + 1) * (self.ny + 1))

        # Element node connectivity verification
        for e in range(self.mesh.nelements):
            nodes = self.mesh.element_nodes[e]
            checks.append(len(nodes) == 4)  # Each element should have 4 nodes

        return all(checks)


# ---------------------- INTEGRATION TESTING FUNCTIONS -----------------------------------------------------------
# **************************************************************************************************************

def run_plate_analysis(shapefile_path: str, material_config: dict, force_configs: list,
                       mesh_params: dict = None, visualization_params: dict = None, save_path: str = None) -> dict:
    """
    Run a complete plate bending analysis with specified configurations.

    This function provides end-to-end testing capabilities by encapsulating the entire
    workflow from mesh creation to results visualization. It enables testing of different
    scenarios while maintaining consistent analysis parameters.

    Parameters:
        shapefile_path (str): Path to the shapefile defining the analysis region
        material_config (dict): Configuration for material properties including pattern_type,
                               material values or ranges, and pattern-specific parameters
        force_configs (list): List of force configurations with coordinates, magnitude,
                             radius, and distribution parameters
        mesh_params (dict, optional): Mesh generation parameters including buffer_distance,
                                     mesh_resolution, and boundary_conditions
        visualization_params (dict, optional): Parameters controlling visualization output
        save_path (str, optional): Directory path for saving visualizations

    Returns:
        dict: Contains mesh object, displacement solution vector, and generated figures
    """
    from social_characteristics_translation import generate_test_layers

    # Set default parameters if not provided
    if mesh_params is None:
        mesh_params = {
            'buffer_distance': 150_000,
            'mesh_resolution': 10_000,
            'boundary_conditions': 'clamped_all',
        }

    if visualization_params is None:
        visualization_params = {
            'plot_materials': True,
            'plot_displacement': True,
            'plot_stress': True,
            'plot_3d': True,
            'view_angle': (20, 270)
        }

    saved_files = []

    def get_figure_from_result(plot_result):
        """
        Extract the figure object from various possible plotting results.

        Parameters:
            plot_result: The result from a plotting function, either a figure directly or a tuple

        Returns:
            matplotlib.figure.Figure or None: The extracted figure object
        """
        if plot_result is None:
            return None
        elif isinstance(plot_result, tuple):
            # If it's a tuple, assume the figure is the first element
            return plot_result[0]
        else:
            # If it's not a tuple, assume it's the figure directly
            return plot_result

    def save_figure(plot_result, base_name: str):
        """
        Save figure with appropriate filename if save_path is specified.

        Parameters:
            plot_result: The plotting result to save
            base_name (str): Base name for the output file
        """
        if save_path is None:
            return

        fig = get_figure_from_result(plot_result)
        if fig is None:
            print(f"Warning: No figure available for {base_name}, skipping save")
            return

        import os
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Generate and use full file path
        filename = generate_filename(base_name)
        filepath = os.path.join(save_path, filename)

        try:
            # Save with high DPI for quality
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"Saved visualization to: {filepath}")
        except Exception as e:
            print(f"Warning: Failed to save {base_name} figure: {str(e)}")

    def generate_filename(base_name: str) -> str:
        """
        Generate a descriptive filename based on physical parameters.

        Returns:
            str: Descriptive filename including material and force parameters
        """
        pattern_type = material_config['pattern_type']

        def format_E(E_val):
            return f"{E_val / 1e9:.4f}"

        # Extract material properties based on pattern type
        if pattern_type == 'uniform':
            E_val = material_config['E']
            nu_val = material_config['nu']
            thickness_val = material_config['thickness']
            # Convert E to GPa for readability
            E_str = f"E{E_val / 1e9:.2f}GPa"
            nu_str = f"nu{nu_val:.2f}"
            thickness_str = f"thickness{thickness_val:.2f}m"

        elif pattern_type == 'split':
            # For split patterns, describe both regions
            if 'north_properties' in material_config:  # North-South split
                props1 = material_config['north_properties']
                props2 = material_config['south_properties']
                region_str = "NS"
            else:  # East-West split
                props1 = material_config['east_properties']
                props2 = material_config['west_properties']
                region_str = "EW"

            # Create property strings showing the range
            E_str = f"E{format_E(props1['E'])}-{format_E(props2['E'])}GPa"
            nu_str = f"nu{props1['nu']:.2f}-{props2['nu']:.2f}"
            thickness_str = f"h{props1['thickness']:.0f}-{props2['thickness']:.0f}m"
            pattern_str = f"split{region_str}"

        elif pattern_type == 'center_zone':
            # For center zone patterns, describe center and outer properties
            center = material_config['center_properties']
            outer = material_config['outer_properties']
            radius = material_config['center_radius']

            E_str = f"E{format_E(center['E'])}-{format_E(outer['E'])}GPa"
            nu_str = f"nu{center['nu']:.2f}-{outer['nu']:.2f}"
            thickness_str = f"h{center['thickness']:.0f}-{outer['thickness']:.0f}m"
            pattern_str = f"center_r{radius:.1f}"

        elif pattern_type in ['gradient_x', 'gradient_y']:
            # For gradient patterns, show the range of values
            E_range = material_config['E_range']
            nu_range = material_config['nu_range']
            thickness_range = material_config['thickness_range']

            E_str = f"E{format_E(E_range[0])}-{format_E(E_range[1])}GPa"
            nu_str = f"nu{nu_range[0]:.2f}-{nu_range[1]:.2f}"
            thickness_str = f"h{thickness_range[0]:.0f}-{thickness_range[1]:.0f}m"
            pattern_str = f"grad_{pattern_type[-1]}"  # 'x' or 'y'
        else:
            # For pattern-based properties, use range
            E_range = material_config['E_range']
            nu_range = material_config['nu_range']
            thickness_range = material_config['thickness_range']
            # Convert E range to GPa for readability
            E_str = f"E{E_range[0] / 1e9:.0f}-{E_range[1] / 1e9:.4f}GPa"
            nu_str = f"nu{nu_range[0]:.2f}-{nu_range[1]:.2f}"
            thickness_str = f"thickness{thickness_range[0]:.2f}-{thickness_range[1]:.2f}m"

        # Handle force description
        if len(force_configs) == 1:
            # Single force case
            force = force_configs[0]
            if 'magnitude' not in force:
                raise ValueError("Force configuration must specify 'magnitude' and 'distribution'")
            force_str = f"F{force['magnitude']:.0f}N"
            if 'distribution' in force:
                force_str += f"_distr-{force['distribution']}_r-{force['radius']:.4f}"
        else:
            # Multiple forces case - create a description for each force
            force_descriptions = []

            for idx, force in enumerate(force_configs, 1):
                if 'magnitude' not in force:
                    raise ValueError(f"Force configuration {idx} must specify 'magnitude'")

                # Create force description including magnitude
                force_desc = f"F{idx}-{force['magnitude']:.0f}N"

                # Add distribution and radius if present
                if 'distribution' in force and 'radius' in force:
                    force_desc += f"_distr-{force['distribution']}_r-{force['radius']:.4f}"

                force_descriptions.append(force_desc)

            # Join all force descriptions
            force_str = '_'.join(force_descriptions)

        # Combine parameters into filename
        params = [thickness_str, E_str, nu_str, force_str]
        filename = f"{base_name}_{'_'.join(params)}.png"
        return filename

    # Initialize mesh
    mesh = FEAMesh2D(width=1.0, height=1.0, nx=1, ny=1)

    # Set up region from shapefile
    mesh._set_region_from_shapefile(
        shapefile_path=shapefile_path,
        buffer_distance=mesh_params['buffer_distance'],
        mesh_resolution=mesh_params['mesh_resolution'],
        visualize=False
    )

    # Generate and apply material properties
    layers = generate_test_layers(mesh, material_config)
    E_combined, nu_combined, thickness_combined = layers._combine_properties('weighted_sum',
                                                                             weights={'test_pattern': 1})
    mesh.E = E_combined
    mesh.nu = nu_combined
    mesh.thickness = thickness_combined

    # Create element mapping and set boundary conditions
    mesh._create_element_mapping()
    fixed_nodes = mesh._get_boundary_nodes(mesh_params['boundary_conditions'])

    # Store generated figures
    figures = {}

    # Visualize material properties if requested
    if visualization_params['plot_materials']:
        fig = mesh._plot_materials()
        figures['materials'] = fig
        save_figure(fig, 'material_properties')

    # Solve the system
    u = apply_multiple_forces(mesh, force_configs, fixed_nodes)

    # Generate requested visualizations
    if visualization_params['plot_displacement']:
        # 2D displacement plot
        fig = mesh._plot_results(u, plot_type='displacement')
        figures['displacement_2d'] = fig
        save_figure(fig, 'displacement_2d')

        if visualization_params['plot_3d']:
            # 3D displacement plot
            fig, ax = mesh._plot_results(u, plot_type='displacement', view_3d=True)
            ax.view_init(elev=visualization_params['view_angle'][0],
                         azim=visualization_params['view_angle'][1])
            figures['displacement_3d'] = (fig, ax)
            save_figure(fig, 'displacement_3d')

    if visualization_params['plot_stress']:
        # 2D stress plot
        fig = mesh._plot_results(u, plot_type='stress', result_type='von_mises')
        figures['stress_2d'] = fig
        save_figure(fig, 'stress_2d')

        if visualization_params['plot_3d']:
            # 3D stress plot
            fig, ax = mesh._plot_results(u, plot_type='stress',
                                         result_type='von_mises', view_3d=True)
            ax.view_init(elev=visualization_params['view_angle'][0],
                         azim=visualization_params['view_angle'][1])
            figures['stress_3d'] = (fig, ax)
            save_figure(fig, 'stress_3d')

    # Calculate and report metrics
    metrics = mesh.get_test_metrics(u)

    print(f"Max displacement: {metrics['max_displacement']:.4e} m")
    print(f"Affected area: {metrics['affected_area_percent']:.2f}%")
    print(f"Max von Mises stress: {metrics['max_von_mises_stress']:.4e} Pa")

    return {
        'mesh': mesh,
        'displacement': u,
        'figures': figures,
        'metrics': metrics
    }


# ---------------------- UTILITY FUNCTIONS ------------------------------------------------------------------------
# **************************************************************************************************************

def run_fea_validation():
    """
    Execute the complete FEA validation suite.

    This function creates a test instance and runs all verification testing,
    providing a comprehensive assessment of the FEA implementation.

    Returns:
        bool: True if all testing pass, False if any test fails
    """
    tester = FEATests()
    return tester.run_all_tests()


def run_integration_test(shapefile_path: str, test_name: str = "Integration Test"):
    """
    Run a standard integration test using the plate analysis function.

    This function provides a convenient way to test the complete workflow
    with predefined configurations.

    Parameters:
        shapefile_path (str): Path to the shapefile for analysis region
        test_name (str): Name identifier for the test

    Returns:
        dict: Results from the plate analysis including success metrics
    """
    # Define standard test configuration
    material_config = {
        'pattern_type': 'uniform',
        'E': 200e9,  # Pa
        'nu': 0.3,
        'thickness': 5000  # meters
    }

    force_configs = [{
        'coordinates': (8.0, 9.0),  # Example coordinates
        'magnitude': 1e12,  # N
        'radius': 50000,  # meters
        'distribution': 'gaussian'
    }]

    mesh_params = {
        'buffer_distance': 100_000,
        'mesh_resolution': 20_000,
        'boundary_conditions': 'clamped_all'
    }

    visualization_params = {
        'plot_materials': False,  # Skip visualizations for automated testing
        'plot_displacement': False,
        'plot_stress': False,
        'plot_3d': False
    }

    print(f"\nRunning {test_name}...")

    try:
        results = run_plate_analysis(
            shapefile_path=shapefile_path,
            material_config=material_config,
            force_configs=force_configs,
            mesh_params=mesh_params,
            visualization_params=visualization_params
        )

        # Check if results are reasonable
        metrics = results['metrics']

        # Basic sanity checks
        checks = []
        checks.append(metrics['max_displacement'] > 0)  # Should have some displacement
        checks.append(metrics['max_displacement'] < 1e6)  # But not unreasonably large
        checks.append(metrics['max_von_mises_stress'] > 0)  # Should have some stress
        checks.append(0 <= metrics['affected_area_percent'] <= 100)  # Percentage should be valid

        success = all(checks)

        print(f"{test_name}: {'PASS' if success else 'FAIL'}")
        if success:
            print(f"  Max displacement: {metrics['max_displacement']:.2e} m")
            print(f"  Max stress: {metrics['max_von_mises_stress']:.2e} Pa")
            print(f"  Affected area: {metrics['affected_area_percent']:.1f}%")

        return {
            'success': success,
            'results': results,
            'metrics': metrics
        }

    except Exception as e:
        print(f"{test_name}: FAIL (Error: {str(e)})")
        return {
            'success': False,
            'error': str(e)
        }