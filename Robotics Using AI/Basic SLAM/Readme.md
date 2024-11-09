# What is SLAM [Simultaneous Localization and Mapping] ?
**Simultaneous Localization and Mapping (SLAM) is a technology used in robotics and autonomous systems to create a map of an unknown environment while simultaneously determining the robot's location within that environment. This enables autonomous navigation without relying on pre-mapped data.**

**Example:** In autonomous vacuum cleaners, SLAM allows the robot to efficiently map and clean a room by recognizing obstacles, furniture, and open spaces in real-time. This ensures thorough cleaning by systematically covering the entire area while avoiding collisions and re-routing when necessary.

**SLAM is integral in applications like self-driving cars, drones, and AR/VR systems for accurate, real-time spatial awareness and navigation. 🚀**

# SLAM Implementation

This repository contains a Python implementation of [Simultaneous Localization and Mapping (SLAM)](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) enhanced with machine learning capabilities, including Extended Kalman Filter (EKF) and neural network-based landmark classification.

## Features

- Basic SLAM implementation with robot movement and landmark detection
- [Extended Kalman Filter (EKF)](https://en.wikipedia.org/wiki/Extended_Kalman_filter) for state estimation
- Neural network-based landmark classification
- Particle filter for robust pose estimation
- Probabilistic sensor modeling
- Visualization of robot path and landmarks

## Prerequisites

Make sure you have [Python](https://www.python.org/) 3.6+ installed. Install the required dependencies:

```bash
pip install numpy matplotlib torch scipyl
```

## Usage
- **Run the simulation:**

```bash
from slam import AdvancedSLAM

# Create and run advanced SLAM simulation
slam = AdvancedSLAM(num_particles=100)  # Initialize with 100 particles
slam.add_landmarks(10)
slam.run_simulation(100)
slam.plot_results()
```

## Parameters

- num_particles: Number of particles for particle filter (default: 100)
- noise: Movement and sensor noise level (default: 0.1)
- max_range: Maximum sensor range for landmark detection (default: 5.0)
- area_size: Size of the simulation area (default: 10)


**Made with ❤️ by [Dinmay](https://github.com/dino65-dev)**