"""
File operations for the application
"""
import json


class FileIO:
    """Handles file input/output operations"""
    
    @staticmethod
    def save_to_file(data, filename):
        """Save data to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Data saved to {filename}.")
    
    @staticmethod
    def load_from_file(filename):
        """Load data from a JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data loaded from {filename}.")
        return data