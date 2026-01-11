"""
Forward modeling operations for geological model
"""
import numpy as np
import pygimli as pg
from pygimli.physics import TravelTimeManager


class ForwardModel:
    """Handles forward modeling for the geological model"""
    
    def __init__(self):
        self.ttmgr = None
        self.mesh = None
        self.response = None
    
    def run_model(self, bodies, points, lines, scheme):
        """Run the forward model with the current geological model.
        
        Args:
            bodies: List of body dictionaries with polygons and velocities
            points: Point manager points
            lines: Line manager lines
            scheme: DataContainer with shot/receiver geometry
            
        Returns:
            Calculated travel times for all shots
        """
        # Store for later use
        self.bodies = bodies
        self.points = points
        self.lines = lines
        self.scheme = scheme
        
        # Create mesh from geological model
        self.mesh = self._create_mesh_from_model(bodies, points, lines, scheme)
        
        # Extract velocities for each body/region
        self.velocity_model = self._create_velocity_model(bodies, self.mesh)
        
        # Run simulation using TravelTimeManager
        self.ttmgr = TravelTimeManager()
        
        # Set data explicitly
        self.ttmgr.fop.setData(scheme)
        
        # Apply the mesh
        self.ttmgr.applyMesh(self.mesh, secNodes=2, ignoreRegionManager=True)
        
        # Calculate response
        slowness = 1.0 / self.velocity_model
        data = self.ttmgr.fop.response(slowness)
        
        # Convert response to numpy array (it's an RVector)
        self.response = np.array(data)
        
        # Extract and store ray paths for later plotting using PyGIMLi's method
        # Note: Ray extraction from forward model may not work with fop.response()
        # Rays are typically only available after inversion
        self.ray_paths = []  # List of ray paths, each ray is a list of (x, y) points
        try:
            # Get ray paths from the manager
            rays = self.ttmgr.getRayPaths()
            # Convert to our format (list of (x, y) tuples)
            for ray in rays:
                ray_x = [pos[0] for pos in ray]
                ray_y = [pos[1] for pos in ray]
                if len(ray_x) > 0:
                    self.ray_paths.append((ray_x, ray_y))
            if len(self.ray_paths) > 0:
                print(f"Extracted {len(self.ray_paths)} ray paths from forward model")
        except Exception:
            # Ray extraction failed - this is expected for forward model
            # Rays are only stored during inversion process
            self.ray_paths = []
            print("Note: Ray paths not available for forward model (only available after inversion)")
        
        return self.response
    
    def plot_mesh(self):
        """Plot the mesh with velocity model and markers."""
        import matplotlib.pyplot as plt
        import pygimli as pg
        
        if self.mesh is None:
            print("No mesh to plot. Run forward model first.")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mesh with markers
        pg.show(self.mesh, markers=True, ax=axes[0], showMesh=True)
        axes[0].set_title("Mesh with Markers")
        axes[0].set_xlabel("Distance (m)")
        axes[0].set_ylabel("Depth (m)")
        
        # Plot velocity model
        if self.velocity_model is not None:
            pg.show(self.mesh, data=self.velocity_model, ax=axes[1], 
                   label="Velocity (m/s)", showMesh=False, cMap="jet")
            axes[1].set_title("Velocity Model")
            axes[1].set_xlabel("Distance (m)")
            axes[1].set_ylabel("Depth (m)")
        
        plt.tight_layout()
        plt.show()
        fig.savefig("forward_model_mesh.png", dpi=300)
    
    def _create_mesh_from_model(self, bodies, points, lines, scheme):
        """Create pygimli mesh from model geometry using polygon merging."""
        import pygimli.meshtools as mt
        
        # Create polygon for each body
        polygons = []
        for i, body in enumerate(bodies):
            poly_points = self._get_body_polygon(body, points, lines)
            if len(poly_points) >= 3:
                # Create closed polygon with body index as marker
                poly = mt.createPolygon(poly_points, isClosed=True, marker=i+1)
                polygons.append(poly)
        
        if not polygons:
            raise ValueError("No valid body polygons created")
        
        # Merge all polygons into a single geometry
        geom = polygons[0]
        for poly in polygons[1:]:
            geom += poly
        
        # Create mesh from merged geometry
        mesh = mt.createMesh(geom, quality=32, area=1.0, smooth=[1, 10])
        
        print(f"Mesh created: {mesh.cellCount()} cells from {len(polygons)} body polygons")
        
        # Associate cells with bodies (handles overlaps and edge cases)
        self._associate_regions(mesh, bodies, points, lines)
        
        return mesh
    
    def _get_body_polygon(self, body, points, lines):
        """Extract polygon vertices from body definition."""
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
    
    def _associate_regions(self, mesh, bodies, points, lines):
        """Associate mesh cells with body markers using point-in-polygon test."""
        from matplotlib.path import Path
        
        # Build paths for all bodies
        body_paths = []
        for i, body in enumerate(bodies):
            poly_points = self._get_body_polygon(body, points, lines)
            if len(poly_points) >= 3:
                body_paths.append((i, Path(poly_points)))
        
        n_bodies = len(bodies)
        background_marker = n_bodies + 1
        
        # Test each cell
        for cell in mesh.cells():
            center = cell.center()
            cell_pt = [center.x(), center.y()]
            
            # Find which body contains this cell (last match wins for overlaps)
            found = False
            for body_idx, path in body_paths:
                if path.contains_point(cell_pt):
                    cell.setMarker(body_idx + 1)
                    found = True
            
            if not found:
                cell.setMarker(background_marker)
    
    def _create_velocity_model(self, bodies, mesh):
        """Create velocity model from body properties and markers."""
        # Get markers for all cells
        vp = np.array(mesh.cellMarkers(), dtype=float)
        
        # Replace marker indices with actual velocities
        for i, body in enumerate(bodies):
            body_marker = i + 1
            vp[vp == body_marker] = body["props"][0]
        
        # Handle background cells (marker = n_bodies + 1)
        # Use velocity from first body as background
        background_marker = len(bodies) + 1
        if len(bodies) > 0:
            vp[vp == background_marker] = bodies[0]["props"][0]
        
        return vp