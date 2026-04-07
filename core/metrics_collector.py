import psutil
import os
import time
import platform
import docker
import string
import random

class MetricsCollector:
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except:
            self.docker_client = None

    def get_system_metrics(self):
        # Provide fallback logic for Windows without os.getloadavg
        try:
            load_avg = os.getloadavg()[0]
        except AttributeError:
            load_avg = psutil.cpu_percent(interval=None) / 100.0

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "cpu_count": psutil.cpu_count(),
            "load_average": load_avg,
            "timestamp": time.time()
        }
    
    def get_per_cpu_metrics(self):
        return psutil.cpu_percent(percpu=True, interval=None)
    
    def get_container_metrics(self, container_id=None):
        metrics = []
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                if container_id:
                    containers = [c for c in containers if c.id.startswith(container_id)]
                for c in containers:
                    metrics.append(self.parse_docker_stats(c))
                return metrics
            except Exception:
                pass
        
        # Fallback: Treat CPU threads or random synthetic blocks as containers
        return self.get_thread_metrics()

    def parse_docker_stats(self, container):
        try:
            stats = container.stats(stream=False)
            # Docker calculate CPU percent
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            number_cpus = stats['cpu_stats']['online_cpus']
            cpu_percent = 0.0
            if system_cpu_delta > 0.0 and cpu_delta > 0.0:
                cpu_percent = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
            
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 1)
            memory_percent = (memory_usage / memory_limit) * 100.0

            return {
                "container_id": container.id[:12],
                "name": container.name,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "time": time.time()
            }
        except:
            return {
                "container_id": container.id[:12],
                "name": container.name,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "time": time.time()
            }

    def get_thread_metrics(self):
        # Simulated fallback for Windows/systems without docker running
        res = []
        per_cpu = psutil.cpu_percent(percpu=True, interval=None)
        for i, perc in enumerate(per_cpu[:4]): # Max 4 to simulate 4 containers
            res.append({
                "container_id": ''.join(random.choices(string.ascii_lowercase + string.digits, k=12)),
                "name": f"synthetic_container_{i}",
                "cpu_percent": perc,
                "memory_percent": random.uniform(10.0, 50.0),
                "time": time.time()
            })
        return res
