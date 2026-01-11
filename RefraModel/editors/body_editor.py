"""
Body editing operations for geological model
"""
from ..geometry.bodies import BodyManager


class BodyEditor:
    """Handles editing of geological bodies"""
    
    def __init__(self):
        self.body_manager = BodyManager()
    
    def edit_body(self, body_index, new_properties):
        """Edit properties of a specific body"""
        if 0 <= body_index < len(self.body_manager.bodies):
            body = self.body_manager.bodies[body_index]
            body.update(new_properties)
            print(f"Updated body {body_index} with properties: {new_properties}")
        else:
            print("Body index out of range.")