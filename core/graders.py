from typing import List

class Grader:
    def __init__(self):
        pass

    def evaluate_task1(self, history: List[dict]):
        # Single Container Stability
        # history holds states 
        if not history: return 0.0
        steps_in_range = 0
        for step_data in history:
            containers = step_data.get('containers', [])
            if containers:
                cpu = containers[0].get('cpu_percent', 0)
                if 65 <= cpu <= 85:
                    steps_in_range += 1
        return steps_in_range / len(history)

    def evaluate_task2(self, history: List[dict]):
        # Multi-Container Priority
        if not history: return 0.0
        critical_health_steps = 0
        priority_compliance = 0
        balance_score = 0.5
        
        for step_data in history:
            containers = step_data.get('containers', [])
            criticals = [c for c in containers if c.get('priority') == 1]
            if criticals:
                if criticals[0].get('is_healthy'):
                    critical_health_steps += 1
                
                # proxy for getting resources first
                if criticals[0].get('current_cpu_shares') > 512:
                    priority_compliance += 1
        
        critical_health_score = critical_health_steps / len(history)
        priority_comp_score = priority_compliance / len(history)
        score = (critical_health_score * 0.5 + priority_comp_score * 0.3 + balance_score * 0.2)
        return min(max(score, 0.0), 1.0)

    def evaluate_task3(self, history: List[dict]):
        # Spike Prediction + Recovery
        if not history: return 0.0
        
        crashes = 0
        total_spikes = 0
        actions_before_spike = 0
        
        for i, step_data in enumerate(history):
            containers = step_data.get('containers', [])
            for c in containers:
                if not c.get('is_healthy'):
                    crashes += 1
                if c.get('cpu_percent', 0) > 90:
                    total_spikes += 1
                    # check previous actions
                    if i > 0:
                        prev_action = history[i-1].get('action')
                        if prev_action and prev_action.get('container_id') == c.get('container_id'):
                            if prev_action.get('cpu_shares_delta', 0) > 0:
                                actions_before_spike += 1

        zero_crash_score = max(1.0 - (crashes * 0.5), 0.0)
        prediction_score = (actions_before_spike / total_spikes) if total_spikes > 0 else 1.0
        recovery_score = 0.8
        cost_score = 0.7
        
        score = (zero_crash_score * 0.4 + prediction_score * 0.3 + recovery_score * 0.2 + cost_score * 0.1)
        return min(max(score, 0.0), 1.0)
