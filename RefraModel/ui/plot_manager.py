"""
Plot management for the application
"""
import matplotlib.pyplot as plt


class PlotManager:
    """Handles plotting operations for the application"""
    
    def __init__(self):
        pass  # Initialize any necessary parameters
    
    def plot_data(self, data):
        """Plot the given data"""
        plt.plot(data)
        plt.title("Data Plot")
        plt.show()