"""
Property editing operations for geological model
"""
from ..geometry.bodies import BodyManager


class PropertyEditor:
    """Handles editing of properties for geological bodies"""
    
    def __init__(self):
        self.body_manager = BodyManager()
    
    def edit_property(self, body_index, property_name, new_value):
        """Edit a specific property of a body"""
        if 0 <= body_index < len(self.body_manager.bodies):
            body = self.body_manager.bodies[body_index]
            if property_name in body["prop_names"]:
                prop_index = body["prop_names"].index(property_name)
                body["props"][prop_index] = new_value
                print(f"Updated property '{property_name}' of body {body_index} to {new_value}")
            else:
                print(f"Property '{property_name}' not found in body {body_index}.")
        else:
            print("Body index out of range.")