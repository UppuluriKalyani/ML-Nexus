import matplotlib.pyplot as plt
import os
from src.config import Config
import numpy as np

class Visualizer:
    def __init__(self, paths, config=Config):
        self.paths = paths
        self.config = config
        
    def plot_price_paths(self, n_paths=50):
        """Plot a subset of price paths"""
        if not os.path.exists(self.config.plots_dir):
            os.makedirs(self.config.plots_dir)
            
        plt.figure(figsize=(12, 6))
        time_points = np.linspace(0, self.config.T, self.config.n_steps + 1)
        
        for i in range(min(n_paths, len(self.paths))):
            plt.plot(time_points, self.paths[i], alpha=0.1, color='blue')
            
        plt.plot(time_points, np.mean(self.paths, axis=0), 'r--', linewidth=2, label='Mean Path')
        plt.axhline(y=self.config.K, color='g', linestyle=':', label='Strike Price')
        
        plt.title('Monte Carlo Simulation: Stock Price Paths')
        plt.xlabel('Time (years)')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.config.plots_dir, 'price_paths.png'))
        plt.close()
