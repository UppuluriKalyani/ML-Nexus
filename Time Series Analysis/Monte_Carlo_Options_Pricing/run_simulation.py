# run_simulation.py
from src.monte_carlo import MonteCarloSimulator
from src.visualization import Visualizer
from src.utils import create_directories

def main():
    # Create necessary directories
    create_directories()
    
    # Initialize and run simulation
    simulator = MonteCarloSimulator()
    paths = simulator.generate_paths()
    
    # Save numerical results
    simulator.save_results()
    
    # Create visualization
    visualizer = Visualizer(paths)
    visualizer.plot_price_paths()
    
    print("Simulation completed! Check the results/ directory for outputs.")

if __name__ == "__main__":
    main()