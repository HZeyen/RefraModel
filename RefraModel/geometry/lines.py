"""
Line management for geological model
"""


class LineManager:
    """Manages lines connecting points"""
    
    def __init__(self):
        self.lines = []
        self.nline = -1
    
    def append_line(self, pt1, pt2, point_manager, body=None):
        """Append a line to the list"""
        self.nline += 1
        p1 = point_manager.points[pt1]
        p2 = point_manager.points[pt2]
        line = {
            "point1": pt1,
            "point2": pt2,
            "bodies": body or [],
            "topo": p1["topo"] and p2["topo"],
            "bottom": p1["bottom"] and p2["bottom"],
            "left": p1["left"] and p2["left"],
            "right": p1["right"] and p2["right"]
        }
        
        self.lines.append(line)
    
    def find_nearest_line(self, xp, yp, points):
        """Find line nearest to a point in screen coordinates"""
        lmin = -1
        dmin = 1000000000.
        
        for i, lin in enumerate(self.lines):
            pt = points[lin["point1"]]
            x1 = pt["xscreen"]
            y1 = pt["yscreen"]
            pt = points[lin["point2"]]
            x2 = pt["xscreen"]
            y2 = pt["yscreen"]
            
            xcross, ycross, x_flag = self.crossing(x1, y1, x2, y2, xp, yp)
            if x_flag:
                d = (xp - xcross) ** 2 + (yp - ycross) ** 2
                if d < dmin:
                    dmin = d
                    lmin = i
        
        return lmin
    
    @staticmethod
    def crossing(x1, y1, x2, y2, xp, yp):
        """Calculate crossing point of perpendicular line"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == dy == 0:
            return None, None, False
        
        vx = xp - x1
        vy = yp - y1
        
        t = (vx * dx + vy * dy) / (dx ** 2 + dy ** 2)
        xq = x1 + t * dx
        yq = y1 + t * dy
        
        flag = True
        if dx > dy:
            if xq < min(x1, x2) or xq > max(x1, x2):
                flag = False
        else:
            if yq < min(y1, y2) or yq > max(y1, y2):
                flag = False
        
        return xq, yq, flag
    
    def check_line_exist(self, x, y, points):
        """Check if a line already exists"""
        import numpy as np
        
        for i, lin in enumerate(self.lines):
            x1 = points[lin["point1"]]["x"]
            x2 = points[lin["point2"]]["x"]
            y1 = points[lin["point1"]]["y"]
            y2 = points[lin["point2"]]["y"]
            
            if (np.isclose(x[0], x1) and np.isclose(y[0], y1) and
                    np.isclose(x[1], x2) and np.isclose(y[1], y2)):
                return i, 1
            elif (np.isclose(x[1], x1) and np.isclose(y[1], y1) and
                    np.isclose(x[0], x2) and np.isclose(y[0], y2)):
                return i, -1
        
        return -1, None
    

