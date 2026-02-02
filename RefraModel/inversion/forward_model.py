"""
Forward modeling operations for geological model
"""
import numpy as np
import time
import gmsh
from ttcrpy.tmesh import Mesh2d
import matplotlib.pyplot as plt


class ForwardModel:
    """Handles forward modeling for the geological model"""
    
    def __init__(self, parent):
        self.ttmgr = None
        self.mesh = None
        self.response = None
        self.new_forward = False
        self.parent = parent
    
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
        self.mesh, self.slowness = self.create_ttcr_mesh()
        self.response, self.rays = self.forward_ttcr(self.mesh, self.slowness)
        if self.parent.show_rays:
            self.parent.plot_rays_forward(self.parent.forward_rays)
        self.plot_ttcr_model()
        self.parent.forward_calculated = True
        return self.response, self.rays

    def get_ray_paths_forward_model(self):
        """Calculate ray paths from Dijkstra model.
           Returns a dictionary with the ray paths. The coordinates of one ray
           may be extracted as paths[nray][1][:,0] for the x coordinates and
           paths[nray][1][:,1] for the y-coordinates.
           paths[nray][0] contains the number of points that define the ray"""
        paths = {}
        npath = -1
        so = np.array(self.scheme["s"], dtype=int)
        rec = np.array(self.scheme["g"], dtype=int)
        pos = np.array(self.scheme.sensorPositions())
        for ns, ss in enumerate(so):
            p = rec[ns]
            recNode = self.ttmgr.fop.mesh().findNearestNode([pos[p, 0], pos[p,2]])
            sourceNode = self.ttmgr.fop.mesh().findNearestNode([pos[ss, 0], pos[ss, 1]])

            path = self.ttmgr.fop.dijkstra.shortestPath(sourceNode, recNode)
            points = self.ttmgr.fop.mesh().positions(withSecNodes=True)[path].array()
            npath += 1
            paths[npath] = points
        return paths

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
# Create closed polygon with body index as marker
            if len(poly_points) >= 3:
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
        if len(bodies[0]["props"]) == 1:
            for i, body in enumerate(bodies):
                body_marker = i + 1
                vp[vp == body_marker] = body["props"][0]
        else:
            for i, body in enumerate(bodies):
                body_marker = i + 1
                zc = []
                ic = []
                for icell, v in enumerate(vp):
                    if v != body_marker:
                        continue
                    ic.append(icell)
                    zc.append(mesh.cellCenters()[icell].y())
                zc = np.array(zc, dtype=float)
                ic = np.array(ic, dtype=int)
                zmax = zc.max()
                for iv, icell in enumerate(ic):
                    vp[icell] = body["props"][0] +(zmax-zc[iv])*body["props"][1]
        # Handle background cells (marker = n_bodies + 1)
        # Use velocity from first body as background
        background_marker = len(bodies) + 1
        if len(bodies) > 0:
            vp[vp == background_marker] = bodies[0]["props"][0]
        
        return vp

    def create_ttcr_mesh(self):
        """ Create a gmsh and vtk mesh together with the velocities and
        slownesses in every cell for use with ttcr package"""
        gmsh.initialize()
        gmsh.clear()
        mpoints = []
        for point in self.parent.point_manager.points:
            if point["topo"]:
                msize = 0.5
            elif point["bottom"]:
                msize = 2.
            else:
                msize = 1.
            mpoints.append(gmsh.model.geo.addPoint(point["x"], 0., -point["y"],
                                                   meshSize=msize))
        tag = 0
        for line in self.parent.line_manager.lines:
            tag += 1
            gmsh.model.geo.addLine(mpoints[line["point1"]],
                                   mpoints[line["point2"]], tag)
        mtag = 1000
        btag = 2000
        for body in self.parent.body_manager.bodies:
            lin = []
            for il, line in enumerate(body["lines"]):
                if body["sense"][il] > 0:
                    lin.append(line+1)
                else:
                    lin.append(-(line+1))
            mtag += 1
            btag += 1
            gmsh.model.geo.addCurveLoop(lin, tag=mtag)
            gmsh.model.geo.addPlaneSurface([mtag], tag=btag)
        gmsh.model.geo.synchronize()
        psurf = []
        for itag in range(2001, btag+1):
            psurf.append(gmsh.model.addPhysicalGroup(2, [itag]))
            gmsh.model.setPhysicalName(2, psurf[-1], f"{itag-2001}")
        gmsh.model.mesh.generate(2)

        triangles = []
        slowness = []
        for dim, tag in gmsh.model.getEntities():
            if dim == 2:
                elemTypes, elemTags, elemNodeTags =\
                    gmsh.model.mesh.getElements(dim, tag)
                physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
                name = gmsh.model.getPhysicalName(dim, physicalTags[0])
                b = self.parent.body_manager.bodies[int(name)]
                props = b["props"]
                for n in range(len(elemTags[0])):
                    t = elemNodeTags[0][3*n:(3*n+3)]
                    triangles.append(t)
                    if len(props) == 1:
                        slowness.append(1./props[0])
                    else:
                        z = 0.
                        for i, tg in enumerate(t):
                            node = gmsh.model.mesh.getNode(tg)
                            z -= node[0][2]
                        z /= len(t)
                        v = props[0] + (b["top"]-z)*props[1]
                        slowness.append(1./v)
        self.slowness = np.array(slowness)
        self.triangles = np.array(triangles)

        uniqueTags = np.unique(self.triangles)
        equiv = np.empty((int(1+uniqueTags.max()),))
        nodes = []
        for n, tag in enumerate(uniqueTags):
            equiv[tag] = n
            node = gmsh.model.mesh.getNode(tag)
            nodes.append(node[0])
        for n1 in range(self.triangles.shape[0]):
            for n2 in range(self.triangles.shape[1]):
                self.triangles[n1, n2] = equiv[self.triangles[n1, n2]]
        nodes = np.array(nodes)
        nodes = np.c_[nodes[:, 0], nodes[:, 2]]
        gmsh.finalize()

        mesh = Mesh2d(nodes, self.triangles.astype(np.int64), method='SPM',
                      n_secondary=25)
        mesh.to_vtk({'slowness': self.slowness}, 'forward')
        self.nodes = nodes
        return mesh, self.slowness

    def forward_ttcr(self, mesh, slowness):
        """Do forward calculation using the ttcr package.
           Input:
               mesh: vtk mesh created in create_ttcr_mesh
               slowness: numpy 1D vector with slowness in each cell"""
        s_uid, ins = np.unique(np.asarray(self.scheme["s"], dtype=int), return_index=True)
        r_uid, inr = np.unique(np.asarray(self.scheme["g"], dtype=int), return_index=True)
        sx = np.array(self.scheme.sensors())[:,::2][s_uid]
        shot_numbers = np.array(self.scheme["s"], dtype=int)
        receiver_numbers = np.array(self.scheme["g"], dtype=int)
        response = np.zeros(len(shot_numbers))
        times = []
        rays = []
        start = time.time()
        for isht, s in enumerate(sx):
            index = np.where(shot_numbers == s_uid[isht])
            r_id = receiver_numbers[index]
            rrx = np.array(self.scheme.sensors())[:,::2][r_id]
            tt, ray = mesh.raytrace(np.array([s]), rrx, slowness, return_rays=True)
            times.append(tt)
            for ir in range(len(ray)):
                ray[ir][:,1] *= -1.
                ipos = np.where((shot_numbers == s_uid[isht]) &
                                (receiver_numbers == r_id[ir]))[0][0]
                response[ipos] = tt[ir]
            rays.append(ray)
            
        end = time.time()
        duration = end-start
        print(f"\nForward calculation took {duration} s")
        return response, rays

    def plot_ttcr_model(self):
        """Plot mesh and veolcities of forward model"""
        V = 1./self.slowness

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        # cmap = self.parent.cmap
        tpc = ax.tripcolor(self.nodes[:, 0], self.nodes[:, 1], self.triangles,
                           V, cmap='coolwarm', edgecolors='w')
        cbar = plt.colorbar(tpc, ax=ax)
        cbar.ax.set_ylabel('Velocity', fontsize=14)

        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')

        plt.xlabel('Distance', fontsize=14)
        plt.ylabel('Depth', fontsize=14)
        plt.tight_layout()
# Save plot of forward model to file within the working folder
        plt.savefig("forward_model_mesh.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

