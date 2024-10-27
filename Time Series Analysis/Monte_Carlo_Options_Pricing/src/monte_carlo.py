import numpy as np
from src.config import Config
import os

class MonteCarloSimulator:
    def __init__(self, config=Config):
        self.config = config
        self.dt = self.config.T / self.config.n_steps
        self.paths = None
        
    def generate_paths(self):
        """Generate stock price paths using Geometric Brownian Motion"""
        np.random.seed(42)  # for reproducibility
        
        drift = (self.config.r - 0.5 * self.config.sigma**2) * self.dt
        vol = self.config.sigma * np.sqrt(self.dt)
        
        # Generate random walks
        Z = np.random.normal(0, 1, (self.config.n_simulations, self.config.n_steps))
        daily_returns = np.exp(drift + vol * Z)
        
        # Create price paths
        self.paths = np.zeros((self.config.n_simulations, self.config.n_steps + 1))
        self.paths[:, 0] = self.config.S0
        for t in range(1, self.config.n_steps + 1):
            self.paths[:, t] = self.paths[:, t-1] * daily_returns[:, t-1]
            
        return self.paths
    
    def price_european_option(self, option_type='call'):
        """Calculate European option price"""
        if self.paths is None:
            self.generate_paths()
            
        final_prices = self.paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - self.config.K, 0)
        else:  # put
            payoffs = np.maximum(self.config.K - final_prices, 0)
            
        option_price = np.exp(-self.config.r * self.config.T) * np.mean(payoffs)
        return option_price
    
    def save_results(self):
        """Save simulation results"""
        if not os.path.exists(self.config.data_dir):
            os.makedirs(self.config.data_dir)
            
        # Save final prices and option prices
        results_path = os.path.join(self.config.data_dir, 'simulation_results.txt')
        call_price = self.price_european_option('call')
        put_price = self.price_european_option('put')
        
        with open(results_path, 'w') as f:
            f.write(f"Simulation Parameters:\n")
            f.write(f"Initial Stock Price: {self.config.S0}\n")
            f.write(f"Strike Price: {self.config.K}\n")
            f.write(f"Time to Maturity: {self.config.T} years\n")
            f.write(f"Risk-free Rate: {self.config.r}\n")
            f.write(f"Volatility: {self.config.sigma}\n")
            f.write(f"\nResults:\n")
            f.write(f"European Call Option Price: {call_price:.4f}\n")
            f.write(f"European Put Option Price: {put_price:.4f}\n")
