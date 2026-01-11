"""
Inversion operations for geological model
"""
import os
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import TravelTimeManager
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Inversion:
    """Handles inversion operations for the geological model"""
    
    def __init__(self):
        self.mgr = None
        self.mesh_inv = None
        self.vest = None
        self.inversion_folder = None
    
    def run_inversion(self, params, scheme, bodies, points, lines, forward_model, body_regularization=None):
        """Run the inversion process with given parameters.
        
        Args:
            params: Dictionary with inversion parameters from dialog
            scheme: DataContainer with picks
            bodies: List of body dictionaries
            points: Point manager points
            lines: Line manager lines
            forward_model: ForwardModel instance
            body_regularization: Dictionary mapping body_idx to regularization params (optional)
            lines: Line manager lines
            forward_model: ForwardModel instance with mesh and velocity
        """
        # Create timestamped folder for results
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.inversion_folder = f"inversion_{timestamp}"
        os.makedirs(self.inversion_folder, exist_ok=True)
        
        print(f"Inversion results will be stored in: {self.inversion_folder}")
        
        # Save inversion parameters
        self._save_parameters(params, bodies)
        
        # Remove zero travel times (shot and receiver at same position)
        scheme = self._remove_zero_times(scheme)
        
        # Create fresh TravelTimeManager (important: don't reuse from previous runs)
        # This ensures region vs cell parameterization is properly reset
        self.mgr = TravelTimeManager(scheme)
        print("Created fresh TravelTimeManager")
        
        # Get model depth from geometry
        ymin = min(p["y"] for p in points)
        ymax = 0.0
        xmin = min(p["x"] for p in points)
        xmax = max(p["x"] for p in points)
        
        # Set maximum iterations
        max_iter = params['max_iterations'] if params['max_iterations'] > 0 else 1000
        
        if params['use_actual_model']:
            # Use actual model as starting model
            print("\nCreating mesh with actual model as starting model")
            
            # Create parametric mesh following body boundaries
            plc = mt.createParaMeshPLC(scheme, paraDepth=-ymin, boundary=0)
            plc = self._create_edges(plc, bodies, points, lines)
            self.mesh_inv = mt.createMesh(plc, quality=32.5, area=10)
            self.mesh_inv = self._associate_regions(self.mesh_inv, bodies, points, lines)
            
            # Check if we have any fixed regions in THIS run - this affects parameterization
            if body_regularization is None:
                body_regularization = {}
            
            has_fixed_regions = any(
                ib in body_regularization and body_regularization[ib].get('type') == 'fixed' 
                for ib in range(len(bodies))
            )
            print(f"DEBUG: has_fixed_regions = {has_fixed_regions}, body_regularization keys = {list(body_regularization.keys())}")
            
            # Use cell-based parameterization for proper cell-wise variation
            # We'll handle single velocity and fixed velocity by post-processing
            print("Using cell-based parameterization (one parameter per cell)")
            self.mgr.setMesh(self.mesh_inv)
            self.mgr.fop.setMesh(self.mesh_inv, ignoreRegionManager=True)
            print(f"  After setMesh: mgr.paraDomain has {self.mgr.paraDomain.cellCount()} cells")
            
            # Clear any previous regularization settings
            self.mgr.inv.setRegularization(background=False)
            
            # Set starting model from forward model for each body
            print("\nSetting starting velocities for regions:")
            
            fixed_regions = []  # Track regions with fixed velocities
            for ib, body in enumerate(bodies):
                marker = ib + 1
                velocity = body["props"][0]
                
                # Apply body-specific regularization if specified
                if ib in body_regularization:
                    print(f"  Applying custom regularization for body {ib} (marker {marker})")
                    result = self._apply_body_regularization(marker, velocity, body_regularization[ib])
                    if result is not None:
                        fixed_regions.append(result)
                else:
                    # Default regularization - free inversion with body velocity as starting model
                    print(f"  Region {marker}: velocity = {velocity:.1f} m/s (slowness = {1.0/velocity:.6f} s/m) - FREE")
            
            # Set starting model for each cell
            print("\nSetting cell-based starting model...")
            start_model = pg.Vector(self.mesh_inv.cellCount())
            for i, cell in enumerate(self.mesh_inv.cells()):
                marker = cell.marker()
                # Find the body with this marker
                body_velocity = bodies[marker-1]["props"][0]
                start_model[i] = 1.0 / body_velocity
            self._cell_start_model = start_model
            print(f"  Prepared starting model for {len(start_model)} cells")
            
            # Store fixed regions info for manual application after inversion
            self.fixed_regions_info = fixed_regions if fixed_regions else None
            
            # Setup per-body velocity constraints based on velocity_limit_percent
            velocity_limit_percent = params.get('velocity_limit_percent', 50.0)
            print(f"\nSetting up velocity constraints (±{velocity_limit_percent:.1f}%):")
            
            # Create lower and upper bound vectors for all cells (in slowness)
            lower_bounds = pg.Vector(self.mesh_inv.cellCount())
            upper_bounds = pg.Vector(self.mesh_inv.cellCount())
            
            # Global limits as fallback
            global_min_slow = 1.0 / params['max_velocity']
            global_max_slow = 1.0 / params['min_velocity']
            
            # Track which path each body takes (for debugging)
            body_constraint_sources = {}
            
            for i, cell in enumerate(self.mesh_inv.cells()):
                marker = cell.marker()
                body_idx = marker - 1
                body_velocity = bodies[body_idx]["props"][0]
                
                # Track constraint source for first cell of each body
                if body_idx not in body_constraint_sources:
                    body_constraint_sources[body_idx] = None
                
                # Check if body has specific regularization overrides
                if body_idx in body_regularization:
                    reg_params = body_regularization[body_idx]
                    reg_type = reg_params.get('type', 'free')
                    
                    if reg_type == 'fixed':
                        # Fixed velocity: set both bounds to same value
                        fixed_vel = reg_params.get('velocity', body_velocity)
                        fixed_slow = 1.0 / fixed_vel
                        lower_bounds[i] = fixed_slow
                        upper_bounds[i] = fixed_slow
                        if body_constraint_sources[body_idx] is None:
                            body_constraint_sources[body_idx] = f"FIXED at {fixed_vel:.1f} m/s"
                        continue
                    elif reg_type == 'range':
                        # Explicit range from dialog
                        min_vel = reg_params.get('vmin', body_velocity * 0.9)
                        max_vel = reg_params.get('vmax', body_velocity * 1.1)
                        lower_bounds[i] = 1.0 / max_vel  # slowness is inverse
                        upper_bounds[i] = 1.0 / min_vel
                        if body_constraint_sources[body_idx] is None:
                            body_constraint_sources[body_idx] = f"RANGE [{min_vel:.1f}, {max_vel:.1f}] m/s"
                        continue
                    elif reg_type == 'single':
                        # Single velocity with strong smoothing - use tight bounds
                        target_vel = reg_params.get('velocity', body_velocity)
                        # Allow small variation (±5%) for numerical stability
                        lower_bounds[i] = 1.0 / (target_vel * 1.05)
                        upper_bounds[i] = 1.0 / (target_vel * 0.95)
                        if body_constraint_sources[body_idx] is None:
                            body_constraint_sources[body_idx] = f"SINGLE at {target_vel:.1f} m/s (±5%)"
                        continue
                    # If 'free' type, fall through to default percentage-based limits
                    # But check if dialog specified a custom percentage
                    if 'velocity_limit_percent' in reg_params:
                        custom_percent = reg_params['velocity_limit_percent']
                        if custom_percent > 0:
                            min_vel = body_velocity * (1.0 - custom_percent / 100.0)
                            max_vel = body_velocity * (1.0 + custom_percent / 100.0)
                            lower_bounds[i] = 1.0 / max_vel
                            upper_bounds[i] = 1.0 / min_vel
                            if body_constraint_sources[body_idx] is None:
                                body_constraint_sources[body_idx] = f"FREE with custom ±{custom_percent:.1f}%"
                            continue
                        # else custom_percent == 0, fall through to global limits
                
                # Default: use velocity_limit_percent from main dialog
                if velocity_limit_percent > 0:
                    # Calculate body-specific limits based on percentage
                    min_vel = body_velocity * (1.0 - velocity_limit_percent / 100.0)
                    max_vel = body_velocity * (1.0 + velocity_limit_percent / 100.0)
                    # Don't clamp to global limits - let each body have its full percentage range
                    # This allows high-velocity bodies to vary properly
                    # Convert to slowness (inverse relationship)
                    lower_bounds[i] = 1.0 / max_vel  # slowness is inverse of velocity
                    upper_bounds[i] = 1.0 / min_vel
                    if body_constraint_sources[body_idx] is None:
                        body_constraint_sources[body_idx] = f"DEFAULT ±{velocity_limit_percent:.1f}%"
                else:
                    # velocity_limit_percent == 0: use global limits
                    lower_bounds[i] = global_min_slow
                    upper_bounds[i] = global_max_slow
                    if body_constraint_sources[body_idx] is None:
                        body_constraint_sources[body_idx] = "GLOBAL limits"
            
            # Print constraint sources for each body
            print("\nConstraint sources:")
            for body_idx, source in sorted(body_constraint_sources.items()):
                print(f"  Body {body_idx} (Region {body_idx + 1}): {source}")
            
            # Print summary of constraints
            for marker in range(1, len(bodies) + 1):
                body_idx = marker - 1
                cell_indices = [i for i, cell in enumerate(self.mesh_inv.cells()) if cell.marker() == marker]
                if cell_indices:
                    sample_idx = cell_indices[0]
                    min_vel = 1.0 / upper_bounds[sample_idx]
                    max_vel = 1.0 / lower_bounds[sample_idx]
                    body_velocity = bodies[body_idx]["props"][0]
                    
                    # Check if bounds are very tight (constant velocity)
                    vel_range = max_vel - min_vel
                    percent_range = (vel_range / body_velocity) * 100.0
                    
                    if percent_range < 1.0:
                        print(f"  Region {marker}: {body_velocity:.1f} m/s with bounds [{min_vel:.1f}, {max_vel:.1f}] m/s (±{percent_range:.2f}% - NEARLY CONSTANT!)")
                    else:
                        print(f"  Region {marker}: {body_velocity:.1f} m/s with bounds [{min_vel:.1f}, {max_vel:.1f}] m/s (±{percent_range:.1f}%)")
            
            # Apply per-body limits using setRegularization for each region
            # This will be the ONLY place we call setRegularization with limits
            print("\nApplying velocity constraints per region using setRegularization:")
            for marker in range(1, len(bodies) + 1):
                body_idx = marker - 1
                body_velocity = bodies[body_idx]["props"][0]
                
                # Get limits for this body from first cell with this marker
                cell_indices = [i for i, cell in enumerate(self.mesh_inv.cells()) if cell.marker() == marker]
                if cell_indices:
                    sample_idx = cell_indices[0]
                    min_slow = lower_bounds[sample_idx]
                    max_slow = upper_bounds[sample_idx]
                    min_vel = 1.0 / max_slow
                    max_vel = 1.0 / min_slow
                    
                    # Set per-region limits and starting model
                    self.mgr.inv.setRegularization(marker, limits=[min_slow, max_slow], 
                                                  startModel=1.0/body_velocity)
                    
                    print(f"  Region {marker}: limits=[{min_slow:.6f}, {max_slow:.6f}] s/m ([{min_vel:.1f}, {max_vel:.1f}] m/s)")
            
            # Calculate zWeight for anisotropic smoothing if requested
            # PyGIMLi's geostatistical constraints don't work well with multiple regions,
            # so we approximate anisotropic behavior using zWeight
            z_weight = params['z_smoothing']
            if params.get('anisotropic', False):
                horiz_corr = params.get('horizontal_correlation', 10.0)
                vert_corr = params.get('vertical_correlation', 1.0)
                # Calculate effective zWeight from correlation lengths
                # zWeight controls vertical vs horizontal smoothing
                # Lower zWeight = less vertical smoothing (more anisotropic)
                # zWeight ≈ (vertical_correlation / horizontal_correlation)²
                z_weight = (vert_corr / horiz_corr) ** 2
                print(f"\nAnisotropic smoothing: horizontal={horiz_corr:.1f}m, vertical={vert_corr:.1f}m")
                print(f"  Calculated effective zWeight = {z_weight:.4f}")
                print(f"  (Lower zWeight = less vertical smoothing, more horizontal preference)")
            
            # Plot starting model
            self._plot_starting_model(self.inversion_folder, bodies)
            
            # Run inversion
            print("\nRunning inversion with actual model...")
            
            # Set delta phi stopping criterion directly on inversion object (as percentage)
            dphi_min = params['max_delta_phi']
            self.mgr.inv.setDeltaPhiStop(dphi_min)
            print(f"Set delta phi stop criterion to {dphi_min:.1f}%")
            
            invert_kwargs = {
                'verbose': True,
                'maxIter': max_iter,
                'lam': params['initial_lambda'],
                'lambdaFactor': params['lambda_reduction'],
                'zWeight': z_weight,
                'useGradient': False
            }
            
            # Don't pass limits or startModel via invert_kwargs - already set via setRegularization
            print(f"\nRunning inversion (limits enforced per region via setRegularization)...")
            
            vel_result = self.mgr.invert(**invert_kwargs)
            
            # Extract and store ray paths for later plotting using PyGIMLi's method
            self.ray_paths = []  # List of ray paths, each ray is a list of (x, y) points
            try:
                # Get ray paths from the manager
                rays = self.mgr.getRayPaths()
                print(f"Extracted {len(rays)} ray paths from inversion")
                # Convert to our format (list of (x, y) tuples)
                for ray in rays:
                    ray_x = [pos[0] for pos in ray]
                    ray_y = [pos[1] for pos in ray]
                    if len(ray_x) > 0:
                        self.ray_paths.append((ray_x, ray_y))
            except Exception as e:
                print(f"Could not extract ray paths: {e}")
                self.ray_paths = []
            
            # Inversion result is now complete
            print(f"\nInversion complete:")
            print(f"  mesh_inv has {self.mesh_inv.cellCount()} cells")
            print(f"  mgr.paraDomain has {self.mgr.paraDomain.cellCount()} cells")
            print(f"  vel_result has {len(vel_result)} values")
            print(f"  vel_result range: min={min(vel_result):.6f}, max={max(vel_result):.6f}")
            
            # Note: Velocity limits are enforced DURING inversion via invert_kwargs['limits']
            # No post-processing clipping is done to preserve consistency with calculated travel times
            
            self.vest = vel_result
            
            # Show final velocities for ALL regions (for debugging)
            print("\nFinal velocities by region (all regions):")
            for marker in range(1, 6):  # Regions 1-5
                velocities = []
                for i, cell in enumerate(self.mesh_inv.cells()):
                    if cell.marker() == marker and i < len(self.vest):
                        velocities.append(self.vest[i])
                if velocities:
                    print(f"  Region {marker}: {len(velocities)} cells, min={min(velocities):.1f}, max={max(velocities):.1f}, mean={np.mean(velocities):.1f}, std={np.std(velocities):.1f} m/s")
                    if len(set(velocities)) == 1:
                        print(f"    WARNING: All cells have identical velocity!")
            
            # Show detail for customized regions
            if body_regularization:
                print("\nDetailed velocities for customized regions:")
                for ib in body_regularization.keys():
                    marker = ib + 1
                    velocities = []
                    for i, cell in enumerate(self.mesh_inv.cells()):
                        if cell.marker() == marker and i < len(self.vest):
                            velocities.append(self.vest[i])
                    if velocities:
                        print(f"  Region {marker}: First 5 values: {[f'{v:.1f}' for v in velocities[:5]]}")
        else:
            # Use gradient model as starting model
            print("\nCreating mesh with gradient starting model")
            
            self.vest = self.mgr.invert(
                scheme,
                secNodes=3,
                paraMaxCellSize=5.0,
                zWeight=params['z_smoothing'],
                vTop=params['v_top'],
                vBottom=params['v_bottom'],
                maxIter=max_iter,
                verbose=True,
                paraDepth=-ymin,
                lam=params['initial_lambda'],
                limits=[params['min_velocity'], params['max_velocity']],
                lambdaFactor=params['lambda_reduction']
            )
            self.mesh_inv = self.mgr.paraDomain
        
        print(f"\nInversion completed. Mesh has {self.mesh_inv.cellCount()} cells")
        
        # Create and save plots
        self._plot_travel_times(scheme)
        self._plot_inverted_model(xmin, xmax)
        
        # Save inversion results
        self._save_results(scheme)
        
        print(f"\nInversion finished. Results saved to {self.inversion_folder}")
        
        return self.vest, self.mgr, scheme
    
    def _create_edges(self, plc, bodies, points, lines):
        """Add body boundary edges to PLC for mesh generation."""
        from matplotlib.path import Path
        
        for body in bodies:
            # Get body polygon vertices
            poly_points = []
            for j, line_idx in enumerate(body["lines"]):
                line = lines[line_idx]
                if j == 0:
                    if body["sense"][j] > 0:
                        p = points[line["point1"]]
                        poly_points.append([p["x"], p["y"]])
                
                if body["sense"][j] > 0:
                    p = points[line["point2"]]
                else:
                    p = points[line["point1"]]
                poly_points.append([p["x"], p["y"]])
            
            # Add edges to PLC
            if len(poly_points) >= 3:
                for i in range(len(poly_points) - 1):
                    try:
                        n1 = plc.createNode(poly_points[i])
                        n2 = plc.createNode(poly_points[i + 1])
                        plc.createEdge(n1, n2)
                    except:
                        pass
        
        return plc
    
    def _associate_regions(self, mesh, bodies, points, lines):
        """Assign body markers to mesh cells using directional extension."""
        from matplotlib.path import Path
        import numpy as np
        
        # Build paths for all bodies (original polygons, no extension)
        body_paths = []
        body_bounds = []  # Store bounds for each body
        
        print("\nBuilding body polygons for region assignment:")
        for i, body in enumerate(bodies):
            velocity = body["props"][0] if body["props"] else 0
            print(f"  Body {i} (will be region {i+1}): velocity = {velocity:.1f} m/s")
            
            # Use the same polygon extraction as forward model
            poly_points = self._get_body_polygon(body, points, lines)
            
            if len(poly_points) >= 3:
                body_paths.append((i, Path(poly_points)))
                
                # Calculate body bounds
                poly_array = np.array(poly_points)
                bounds = {
                    'xmin': poly_array[:, 0].min(),
                    'xmax': poly_array[:, 0].max(),
                    'ymin': poly_array[:, 1].min(),
                    'ymax': poly_array[:, 1].max(),
                    'ymean': poly_array[:, 1].mean()
                }
                body_bounds.append(bounds)
        
        # Get mesh bounds
        mesh_xmin = mesh.xmin()
        mesh_xmax = mesh.xmax()
        mesh_ymin = mesh.ymin()
        
        # First pass: assign cells inside bodies
        # Process bodies in reverse order so overlaps favor later bodies
        assigned_markers = {}  # Store which cells are assigned
        
        for cell_idx, cell in enumerate(mesh.cells()):
            center = cell.center()
            cell_pt = [center.x(), center.y()]
            
            # Try to assign to a body (last match wins for overlaps)
            assigned = False
            for body_idx, path in body_paths:
                if path.contains_point(cell_pt):
                    marker = body_idx + 1
                    cell.setMarker(marker)
                    assigned_markers[cell_idx] = marker
                    assigned = True
                    # Don't break - let later bodies override if there's overlap
            
            if not assigned:
                assigned_markers[cell_idx] = None
        
        # Second pass: assign unassigned cells to nearest assigned cell
        unassigned_count = sum(1 for m in assigned_markers.values() if m is None)
        if unassigned_count > 0:
            print(f"\n  Assigning {unassigned_count} cells using nearest assigned cell...")
            
            # Collect positions and markers of all assigned cells
            from scipy.spatial import cKDTree
            assigned_positions = []
            assigned_marker_list = []
            
            for cell_idx, cell in enumerate(mesh.cells()):
                if assigned_markers[cell_idx] is not None:
                    center = cell.center()
                    assigned_positions.append([center.x(), center.y()])
                    assigned_marker_list.append(assigned_markers[cell_idx])
            
            # Build KD-tree for fast nearest neighbor search
            if assigned_positions:
                tree = cKDTree(assigned_positions)
                
                # Assign each unassigned cell to nearest assigned cell's marker
                for cell_idx, cell in enumerate(mesh.cells()):
                    if assigned_markers[cell_idx] is not None:
                        continue  # Skip already assigned cells - DON'T CHANGE THEM!
                    
                    center = cell.center()
                    cell_pt = [center.x(), center.y()]
                    
                    # Find nearest assigned cell
                    _, idx = tree.query(cell_pt)
                    nearest_marker = assigned_marker_list[idx]
                    
                    cell.setMarker(nearest_marker)
                    assigned_markers[cell_idx] = nearest_marker
        
        # Count markers
        marker_counts = {}
        for cell in mesh.cells():
            marker = cell.marker()
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        print("\nRegion assignment in mesh:")
        for marker in sorted(marker_counts.keys()):
            print(f"  Region {marker}: {marker_counts[marker]} cells")
        
        return mesh
    
    def _get_body_polygon(self, body, points, lines):
        """Extract polygon vertices from body definition (same as forward model)."""
        poly_points = []
        
        # Get first point
        line = lines[body["lines"][0]]
        if body["sense"][0] > 0:
            p = points[line["point1"]]
            poly_points.append([p["x"], p["y"]])
        else:
            p = points[line["point2"]]
            poly_points.append([p["x"], p["y"]])
        
        # Walk through all lines adding end points
        for j, line_idx in enumerate(body["lines"]):
            line = lines[line_idx]
            if body["sense"][j] > 0:
                p = points[line["point2"]]
            else:
                p = points[line["point1"]]
            poly_points.append([p["x"], p["y"]])
        
        return poly_points[:-1]  # Remove duplicate closing point
    
    def _apply_body_regularization(self, marker, default_velocity, reg_params):
        """Apply body-specific regularization settings.
        
        Args:
            marker: Region marker number
            default_velocity: Body's default velocity
            reg_params: Dictionary with regularization parameters from dialog
            
        Returns:
            Fixed velocity if type='fixed', None otherwise
        """
        reg_type = reg_params.get('type', 'free')
        slowness = 1.0 / default_velocity
        
        print(f"  Region {marker}: {reg_type} regularization")
        
        if reg_type == 'free':
            # Free inversion - limits will be applied by the main setRegularization loop
            # Just return None so the standard limits calculation is used
            print(f"    Starting velocity: {default_velocity:.1f} m/s (free inversion)")
            return None
            
        elif reg_type == 'single':
            # Single velocity - set all cells to same value after inversion
            target_vel = reg_params.get('velocity', default_velocity)
            target_slowness = 1.0 / target_vel
            print(f"    Target velocity: {target_vel:.1f} m/s (will set all cells to this value)")
            return {'marker': marker, 'velocity': target_vel, 'slowness': target_slowness}
            
        elif reg_type == 'range':
            # Velocity range - apply bounds by post-clipping (setRegularization doesn't work with cell-based)
            min_vel = reg_params.get('vmin', default_velocity * 0.9)
            max_vel = reg_params.get('vmax', default_velocity * 1.1)
            print(f"    Velocity range: {min_vel:.1f} - {max_vel:.1f} m/s (will clip after inversion)")
            return {'marker': marker, 'min_velocity': min_vel, 'max_velocity': max_vel, 'type': 'range'}
            
        elif reg_type == 'fixed':
            # Fixed velocity - store for later processing
            # We'll handle this by setting individual cell constraints after mesh setup
            fixed_vel = reg_params.get('fixed_velocity', default_velocity)
            fixed_slowness = 1.0 / fixed_vel
            # For now, just set the starting model
            self.mgr.inv.setRegularization(marker, startModel=fixed_slowness, background=False,
                                          zWeight=1.0, fix=False)
            print(f"    Fixed velocity: {fixed_vel:.1f} m/s (will constrain individual cells)")
            return {'marker': marker, 'velocity': fixed_vel, 'slowness': fixed_slowness}
        
        # Handle anisotropy if specified
        if reg_params.get('anisotropy', False):
            horiz = reg_params.get('horizontal_correlation', 10.0)
            vert = reg_params.get('vertical_correlation', 1.0)
            # Apply correlation length constraints
            # Note: This would require custom constraint matrix setup
            print(f"    Anisotropic smoothing: horizontal={horiz:.1f}m, vertical={vert:.1f}m")
        
        return None
    
    def _remove_zero_times(self, scheme):
        """Remove data points with zero travel times (shot == receiver position)."""
        # Get travel times
        times = np.array(scheme["t"])
        
        # Find indices of non-zero times
        valid_indices = np.where(times > 0.0)[0]
        
        if len(valid_indices) < len(times):
            print(f"Removing {len(times) - len(valid_indices)} data points with zero travel times")
            
            # Create new scheme with only valid data
            new_scheme = pg.DataContainer()
            new_scheme.registerSensorIndex('s')
            new_scheme.registerSensorIndex('g')
            
            # Copy sensors
            for i in range(scheme.sensorCount()):
                new_scheme.createSensor(scheme.sensorPosition(i))
            
            # Copy only valid data entries
            new_scheme.resize(len(valid_indices))
            new_scheme.set('s', np.array(scheme["s"])[valid_indices])
            new_scheme.set('g', np.array(scheme["g"])[valid_indices])
            new_scheme.set('t', times[valid_indices])
            
            # Copy errors if they exist
            try:
                errors = np.array(scheme["err"])[valid_indices]
                new_scheme.set('err', errors)
            except:
                new_scheme.set('err', np.ones(len(valid_indices)) * 0.0001)
            
            return new_scheme
        
        return scheme
    
    def _save_parameters(self, params, bodies):
        """Save inversion parameters to file."""
        param_file = os.path.join(self.inversion_folder, "inversion_parameters.txt")
        
        with open(param_file, "w") as f:
            f.write("INVERSION PARAMETERS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Starting Model:\n")
            if params['use_actual_model']:
                f.write("   Actual geological model\n")
            else:
                f.write("   Gradient model\n")
                f.write(f"      v_top: {params['v_top']} m/s\n")
                f.write(f"      v_bottom: {params['v_bottom']} m/s\n")
            
            f.write("\nAbort Criteria:\n")
            if params['abort_chi2']:
                f.write("   Chi² <= 1.0\n")
            f.write(f"   Maximum iterations: {params['max_iterations']}\n")
            
            f.write("\nSmoothing Parameters:\n")
            f.write(f"   Initial lambda: {params['initial_lambda']}\n")
            f.write(f"   Lambda reduction factor: {params['lambda_reduction']}\n")
            f.write(f"   Z-direction smoothing: {params['z_smoothing']}\n")
            
            f.write("\nVelocity Constraints:\n")
            f.write(f"   Minimum velocity: {params['min_velocity']} m/s\n")
            f.write(f"   Maximum velocity: {params['max_velocity']} m/s\n")
        
        print(f"Parameters saved to {param_file}")
    
    def _plot_starting_model(self, xmin, xmax):
        """Plot and save the starting model."""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        start_model = 1.0 / self.mgr.inv.fop.startModel()
        self.mgr.inv.fop.paraDomain.show(
            start_model, ax=ax, colorBar=True, 
            logScale=False, label='Velocity [m/s]'
        )
        
        ax.set_xlim(left=xmin, right=xmax)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Depth [m]")
        ax.set_title("Starting Model")
        
        fig.savefig(os.path.join(self.inversion_folder, "starting_model.png"), dpi=150)
        plt.close(fig)
        print("Starting model plot saved")
    
    def _plot_travel_times(self, scheme):
        """Plot measured vs calculated travel times, separated by shot point."""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Calculate chi2 and RMS
        chi2 = self.mgr.inv.chi2()
        calc_tt = np.array(self.mgr.inv.response)
        meas_tt = np.array(scheme["t"])
        rms = np.sqrt(np.mean((meas_tt - calc_tt)**2))
        
        # Get shot positions to identify shot changes
        shot_positions = np.array(scheme['s'])
        
        # Find where shot point changes (new shot begins)
        shot_changes = [0]  # Start of first shot
        for i in range(1, len(shot_positions)):
            if shot_positions[i] != shot_positions[i-1]:
                shot_changes.append(i)
        shot_changes.append(len(shot_positions))  # End of last shot
        
        # Plot each shot separately to avoid connecting different shots
        for i in range(len(shot_changes) - 1):
            start_idx = shot_changes[i]
            end_idx = shot_changes[i + 1]
            indices = np.arange(start_idx, end_idx)
            
            # Plot measured and calculated for this shot
            if i == 0:
                ax.plot(indices, meas_tt[start_idx:end_idx], 'b+', label='Measured', markersize=8)
                ax.plot(indices, calc_tt[start_idx:end_idx], 'r-', label='Calculated', linewidth=1)
            else:
                ax.plot(indices, meas_tt[start_idx:end_idx], 'b+', markersize=8)
                ax.plot(indices, calc_tt[start_idx:end_idx], 'r-', linewidth=1)
        
        ax.set_xlabel("Data point index")
        ax.set_ylabel("Travel time [s]")
        ax.set_title(f"Measured vs Calculated Travel Times\n"
                    f"RMS: {rms*1000:.2f} ms, Chi²: {chi2:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.savefig(os.path.join(self.inversion_folder, "travel_times.png"), dpi=150)
        plt.close(fig)
        print(f"Travel times plot saved (RMS: {rms*1000:.2f} ms, Chi²: {chi2:.3f})")
    
    def _plot_inverted_model(self, xmin, xmax):
        """Plot the inverted velocity model with chi2 history."""
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(100, 100, figure=fig)
        ax_model = fig.add_subplot(gs[:, :80])
        ax_chi = fig.add_subplot(gs[30:70, 90:])
        
        # Plot inverted model - use mesh_inv with actual velocity data
        vmin = np.quantile(self.vest, 0.01)
        vmax = np.quantile(self.vest, 0.99)
        
        # Plot the mesh with actual cell velocities
        pg.show(self.mesh_inv, data=self.vest, ax=ax_model,
                cMin=vmin, cMax=vmax, cMap='jet',
                logScale=False, label='Velocity [m/s]', colorBar=True)
        
        # Draw ray paths
        self.mgr.drawRayPaths(ax=ax_model, color="black", lw=0.3, alpha=0.5)
        
        ax_model.set_xlim(left=xmin, right=xmax)
        ax_model.set_xlabel("Distance [m]")
        ax_model.set_ylabel("Depth [m]")
        ax_model.set_title(f"Inverted Velocity Model\n"
                          f"(vmin={np.min(self.vest):.0f}, vmax={np.max(self.vest):.0f} m/s)")
        
        # Plot chi2 evolution
        chi_history = np.log10(np.array(self.mgr.inv.chi2History))
        iterations = np.arange(len(chi_history)) + 1
        
        ax_chi.plot(iterations, chi_history, 'b-o', markersize=4)
        ax_chi.set_ylabel("log₁₀(Chi²)")
        ax_chi.set_xlabel("Iteration #")
        ax_chi.set_title("Convergence")
        ax_chi.grid(True, alpha=0.3)
        
        fig.savefig(os.path.join(self.inversion_folder, "inverted_model.png"), dpi=150)
        plt.close(fig)
        print("Inverted model plot saved")
    
    def _save_results(self, scheme):
        """Save inversion results to files."""
        # Save velocity model with cell center coordinates
        model_file = os.path.join(self.inversion_folder, "velocity_model.txt")
        with open(model_file, 'w') as f:
            f.write("# Cell_X[m]  Cell_Y[m]  Velocity[m/s]\n")
            for i, cell in enumerate(self.mesh_inv.cells()):
                center = cell.center()
                f.write(f"{center[0]:.6f}  {center[1]:.6f}  {self.vest[i]:.6f}\n")
        
        # Save mesh
        mesh_file = os.path.join(self.inversion_folder, "mesh.bms")
        self.mesh_inv.save(mesh_file)
        
        # Save travel time fit with shot and receiver information
        calc_tt = np.array(self.mgr.inv.response)
        meas_tt = np.array(scheme["t"])
        shot_x = np.array(scheme['s'])  # Shot positions
        recv_x = np.array(scheme['g'])  # Receiver positions
        fit_file = os.path.join(self.inversion_folder, "travel_time_fit.txt")
        
        # Get unique shot and receiver positions to create numbering
        unique_shots = np.unique(shot_x)
        unique_recvs = np.unique(recv_x)
        shot_to_num = {pos: i+1 for i, pos in enumerate(unique_shots)}
        recv_to_num = {pos: i+1 for i, pos in enumerate(unique_recvs)}
        
        with open(fit_file, "w") as f:
            f.write("# Index  Shot#  Recv#  Measured[s]  Calculated[s]  Residual[s]\n")
            for i, (m, c) in enumerate(zip(meas_tt, calc_tt)):
                shot_num = shot_to_num[shot_x[i]]
                recv_num = recv_to_num[recv_x[i]]
                f.write(f"{i:6d}  {shot_num:5d}  {recv_num:5d}  {m:.6f}  {c:.6f}  {m-c:.6f}\n")
        
        print(f"Results saved to {self.inversion_folder}")
    
    def _plot_starting_model(self, output_dir, bodies):
        """Plot starting model with markers and velocities."""
        import matplotlib.pyplot as plt
        import pygimli as pg
        import numpy as np
        
        # Create velocity model from markers
        velocity_model = np.zeros(self.mesh_inv.cellCount())
        for cell_idx, cell in enumerate(self.mesh_inv.cells()):
            marker = cell.marker()
            # Find body with this marker
            if marker > 0 and marker <= len(bodies):
                velocity_model[cell_idx] = bodies[marker - 1]["props"][0]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot mesh with markers
        pg.show(self.mesh_inv, markers=True, ax=axes[0], showMesh=True, 
                label="Region Marker", colorBar=True)
        axes[0].set_title("Starting Model - Region Markers")
        axes[0].set_xlabel("Distance (m)")
        axes[0].set_ylabel("Depth (m)")
        
        # Plot mesh with velocities
        pg.show(self.mesh_inv, data=velocity_model, ax=axes[1], 
                label="Velocity (m/s)", showMesh=False, colorBar=True, cMap="jet")
        axes[1].set_title("Starting Model - Velocities")
        axes[1].set_xlabel("Distance (m)")
        axes[1].set_ylabel("Depth (m)")
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "starting_model.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Starting model plot saved to starting_model.png")