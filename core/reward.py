from statistics import mean
from core.models import ResourceObservation, ResourceAction, ResourceReward

def calculate_reward(
    before: ResourceObservation,
    action: ResourceAction, 
    after: ResourceObservation
) -> ResourceReward:
    
    crashed = [c for c in after.containers if not c.is_healthy]
    critical_degraded = [c for c in after.containers if c.priority == 1 and c.cpu_percent > 90]
    
    perf = 1.0 - (len(crashed) * 0.5) - (len(critical_degraded) * 0.3)
    perf = max(0.0, perf)
    
    if not after.containers:
        eff = 0.5
    else:
        avg_cpu = mean([c.cpu_percent for c in after.containers])
        if 65 <= avg_cpu <= 85:
            eff = 1.0
        elif 50 <= avg_cpu < 65 or 85 < avg_cpu <= 92:
            eff = 0.6
        elif avg_cpu < 50:
            eff = 0.2
        else:
            eff = 0.1
    
    change_mag = abs(action.cpu_shares_delta) / 256.0
    stab = 1.0 - (change_mag * 0.5)
    
    pred_bonus = 0.0
    for c in after.containers:
        if c.container_id == action.container_id and c.cpu_trend > 0.1 and action.cpu_shares_delta > 0:
            pred_bonus += 0.1
    
    total = min(max((perf * 0.50) + (eff * 0.30) + (stab * 0.20) + pred_bonus, 0.0), 1.0)
    
    return ResourceReward(
        total=total,
        performance_score=perf,
        efficiency_score=eff,
        stability_score=stab,
        prediction_bonus=pred_bonus,
        breakdown={
            "performance": perf * 0.50,
            "efficiency": eff * 0.30,
            "stability": stab * 0.20,
            "prediction_bonus": pred_bonus
        }
    )
