"""
Point management for geological model
"""
import numpy as np


class PointManager:
    """Manages points in the model"""
    
    def __init__(self, xmin, xmax, ymin, ymax, eps_x, eps_y):
        self.points = []
        self.npoint = -1
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.eps_x = eps_x
        self.eps_y = eps_y
    
    def append_point(self, xp, yp, xs, ys, body=None, line=None):
        """Append a point to the list of points"""
        self.npoint += 1
        # Round coordinates to 2 decimals at creation
        try:
            xp = float(round(xp, 2))
            yp = float(round(yp, 2))
        except Exception:
            pass
        
        point = {
            "x": xp,
            "y": yp,
            "xscreen": xs,
            "yscreen": ys,
            "lines": line or [],
            "bodies": body or [],
            "topo": False,
            "bottom": False,
            "left": False,
            "right": False
        }
        
        # Check special positions
        if np.isclose(yp, self.ymin):
            point["bottom"] = True
        if xp <= self.xmin + 0.1:
            point["left"] = True
        if xp >= self.xmax - 0.01:
            point["right"] = True
        
        self.points.append(point)
    
    def find_nearest_point(self, xp, yp):
        """Find the nearest point to given coordinates"""
        if not self.points:
            return -1
        
        pmin = -1
        dmin = 1000000000.
        
        for i, p in enumerate(self.points):
            d = (p["x"] - xp) ** 2 + (p["y"] - yp) ** 2
            if d < dmin:
                pmin = i
                dmin = d
        
        return pmin
    
    def find_nearest_screen_point(self, xp, yp):
        """Find nearest point in screen coordinates"""
        if not self.points:
            return -1
        
        pmin = -1
        dmin = 1000000000.
        
        for i, p in enumerate(self.points):
            d = (p["xscreen"] - xp) ** 2 + (p["yscreen"] - yp) ** 2
            if d < dmin:
                pmin = i
                dmin = d
        
        return pmin
    
    def check_close(self, x, y):
        """Check if point is near existing point"""
        if not self.points:
            return x, y, -1
        
        for i, p in enumerate(self.points):
            xb = p["x"]
            yb = p["y"]
            if abs(x - xb) < self.eps_x and abs(y - yb) < self.eps_y:
                return xb, yb, i
        
        return x, y, -1