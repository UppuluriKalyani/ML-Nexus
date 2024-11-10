import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2
import random
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import copy

# Base Robot Class
class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.landmarks = []  
        self.observed = []   
        
    def move(self, distance, turn_angle, noise=0.1):
        distance += random.gauss(0, noise)
        turn_angle += random.gauss(0, noise * 0.1)
        
        self.x += distance * cos(self.theta)
        self.y += distance * sin(self.theta)
        self.theta = (self.theta + turn_angle) % (2 * np.pi)
        
    def sense_landmarks(self, landmarks, max_range=5.0, noise=0.1):
        observations = []
        
        for lm in landmarks:
            dx = lm[0] - self.x
            dy = lm[1] - self.y
            distance = sqrt(dx**2 + dy**2)
            
            if distance <= max_range:
                angle = (atan2(dy, dx) - self.theta) % (2 * np.pi)
                distance += random.gauss(0, noise)
                angle += random.gauss(0, noise * 0.1)
                observations.append((distance, angle, lm))
                
        return observations

# Neural Network for Landmark Classification
class LandmarkClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_classes=2):
        super(LandmarkClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# EKF SLAM Implementation
class EKF_SLAM:
    def __init__(self, state_dim=3, landmark_dim=2):
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        self.Q = np.diag([0.1, 0.1, 0.1])
        self.R = np.diag([0.1, 0.1])
        self.landmarks = {}
        self.landmark_classifier = LandmarkClassifier()
        
    def predict(self, control):
        dx, dy, dtheta = control
        self.state[0] += dx * cos(self.state[2])
        self.state[1] += dy * sin(self.state[2])
        self.state[2] += dtheta
        
        G = np.eye(self.state_dim)
        G[0, 2] = -dx * sin(self.state[2])
        G[1, 2] = dx * cos(self.state[2])
        
        self.covariance = G @ self.covariance @ G.T + self.Q
        
    def update(self, measurement, landmark_id):
        if landmark_id not in self.landmarks:
            r, bearing = measurement
            x = self.state[0] + r * cos(bearing + self.state[2])
            y = self.state[1] + r * sin(bearing + self.state[2])
            self.landmarks[landmark_id] = np.array([x, y])
        
        landmark_pos = self.landmarks[landmark_id]
        dx = landmark_pos[0] - self.state[0]
        dy = landmark_pos[1] - self.state[1]
        q = dx**2 + dy**2
        
        z_hat = np.array([sqrt(q), atan2(dy, dx) - self.state[2]])
        
        H = np.zeros((2, self.state_dim))
        H[0, 0] = -dx/sqrt(q)
        H[0, 1] = -dy/sqrt(q)
        H[1, 0] = dy/q
        H[1, 1] = -dx/q
        H[1, 2] = -1
        
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        innovation = measurement - z_hat
        self.state += K @ innovation
        self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance

# Advanced Robot with EKF and ML capabilities
class AdvancedRobot(Robot):
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        super().__init__(x, y, theta)
        self._prev_x = x
        self._prev_y = y
        self._prev_theta = theta
        self.ekf_slam = EKF_SLAM()
        self.particle_weight = 1.0
        self.sensor_model = multivariate_normal(mean=[0, 0], cov=[[0.1, 0], [0, 0.1]])

    def update_pose_estimate(self, observation):
        control = [self.x - self._prev_x, self.y - self._prev_y, self.theta - self._prev_theta]
        self.ekf_slam.predict(control)
        
        for obs in observation:
            self.ekf_slam.update(obs[:2], obs[2])
        
        self.particle_weight *= self.compute_observation_likelihood(observation)
    
    def compute_observation_likelihood(self, observation):
        likelihood = 1.0
        for obs in observation:
            expected_obs = self.get_expected_observation(obs[2])
            likelihood *= self.sensor_model.pdf(obs[:2] - expected_obs)
        return likelihood

# Advanced SLAM with ML and Particle Filter
class AdvancedSLAM:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = [AdvancedRobot() for _ in range(num_particles)]
        self.landmark_classifier = LandmarkClassifier()
        self.robot_path = []
        self.landmarks = []
        
    def run_simulation(self, steps=100):
        for _ in range(steps):
            for particle in self.particles:
                distance = random.uniform(0.1, 0.5)
                turn_angle = random.uniform(-0.2, 0.2)
                particle.move(distance, turn_angle)
                
            self.res
            
            best_particle = max(self.particles, key=lambda p: p.particle_weight)
            self.robot_path.append((best_particle.x, best_particle.y))
    
    def resample_particles(self):
        weights = [p.particle_weight for p in self.particles]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        new_particles = []
        for _ in range(self.num_particles):
            idx = np.random.choice(len(self.particles), p=weights)
            new_particles.append(copy.deepcopy(self.particles[idx]))
            
        self.particles = new_particles

    def plot_results(self):
        plt.figure(figsize=(10, 10))
        
        # Plot path and landmarks
        path_x = [p[0] for p in self.robot_path]
        path_y = [p[1] for p in self.robot_path]
        plt.plot(path_x, path_y, 'b-', label='Robot Path')
        
        if self.landmarks:
            lm_x = [l[0] for l in self.landmarks]
            lm_y = [l[1] for l in self.landmarks]
            plt.scatter(lm_x, lm_y, c='red', marker='*', s=100, label='Landmarks')
        
        plt.title('Advanced SLAM Simulation')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    slam = AdvancedSLAM(num_particles=100)
    slam.run_simulation(steps=100)
    slam.plot_results()