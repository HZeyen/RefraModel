"""
Node editing operations for geological model
"""
from ..geometry.points import PointManager


class NodeEditor:
    """Handles editing of nodes (points) in the model"""
    
    def __init__(self):
        self.point_manager = PointManager(0, 100, 0, 100, 1, 1)
    
    def edit_node(self, node_index, new_coordinates):
        """Edit coordinates of a specific node"""
        if 0 <= node_index < len(self.point_manager.points):
            point = self.point_manager.points[node_index]
            point["x"], point["y"] = new_coordinates
            print(f"Updated node {node_index} to coordinates: {new_coordinates}")
        else:
            print("Node index out of range.")