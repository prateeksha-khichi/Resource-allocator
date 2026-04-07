import math
import numpy as np

class WorkloadSimulator:
    
    def web_traffic_pattern(self, t):
        daily = 0.3 * math.sin(2 * math.pi * t / 86400)
        spike = np.random.poisson(0.1) * np.random.exponential(0.3)
        noise = np.random.normal(0, 0.05)
        return min(max(0.4 + daily + spike + noise, 0.05), 0.99)
    
    def batch_job_pattern(self, t, duration=3600):
        if t < duration * 0.1: 
            return t / (duration * 0.1) * 0.9
        elif t < duration * 0.9: 
            return 0.9 + np.random.normal(0, 0.02)
        else: 
            return max(0.9 - (t - duration * 0.9) / (duration * 0.1) * 0.9, 0.05)
    
    def ml_training_pattern(self, t, total_epochs=100):
        progress = (t % total_epochs) / total_epochs
        cpu = 0.85 + np.random.normal(0, 0.03)
        memory = 0.3 + 0.6 * progress + np.random.normal(0, 0.02)
        return min(max(cpu, 0.05), 0.99), min(max(memory, 0.1), 0.98)
    
    def microservice_pattern(self, t):
        burst = np.random.exponential(0.05) if np.random.random() < 0.1 else 0
        return min(0.15 + burst, 0.99)
