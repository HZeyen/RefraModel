"""
Body management for geological model
"""
import numpy as np
from matplotlib.path import Path


class BodyManager:
    """Manages geological bodies in the model"""
    
    def __init__(self):
        self.bodies = []
        self.nbody = -1
    
    def append_body(self, lines, directions, para_names, para_values=None, name=None):
        """Add a new body to the list of bodies"""
        self.nbody += 1
        print(f"Body {self.nbody} created")
        
        body = {
            "lines": lines,
            "sense": directions,
            "name": name or f"Body_{self.nbody}",
            "prop_names": para_names,
            "props": para_values or [0.] * len(para_names),
            "cuts": [],
            "constraint": "free",
            "limits": [0., 10000.],
            "corr_lengths": [0., 0.]
        }
        
        self.bodies.append(body)
    
    def get_polygon(self, nb, points, lines):
        """Get polygon coordinates for a body"""
        b = self.bodies[nb]
        line = lines[b["lines"][0]]
        
        if b["sense"][0] > 0:
            x = [points[line["point1"]]["x"]]
            y = [points[line["point1"]]["y"]]
        else:
            x = [points[line["point2"]]["x"]]
            y = [points[line["point2"]]["y"]]
        
        for j, lin in enumerate(b["lines"]):
            line = lines[lin]
            if b["sense"][j] > 0:
                x.append(points[line["point2"]]["x"])
                y.append(points[line["point2"]]["y"])
            else:
                x.append(points[line["point1"]]["x"])
                y.append(points[line["point1"]]["y"])
        
        return x, y
    
    def inside_body(self, xp, yp, points, lines):
        """Check whether a point is inside a body"""
        for i, b in enumerate(self.bodies):
            x, y = self.get_polygon(i, points, lines)
            path = Path(list(zip(x, y)))
            if xp is None:
                return -1, path
            if path.contains_point((xp, yp)):
                return i, path
        return -1, None
    
    def get_vel_limits(self):
        """Calculate minimum and maximum velocities"""
        if not self.bodies:
            return 1000., 5000.
        
        vmin = self.bodies[0]["props"][0]
        vmax = vmin
        for b in self.bodies:
            vmin = min(vmin, b["props"][0])
            vmax = max(vmax, b["props"][0])
        
        if vmin == vmax:
            vmin -= 100.
            vmax += 100.
        
        return vmin, vmax