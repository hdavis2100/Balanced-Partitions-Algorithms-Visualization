from typing import Dict

def max_abs_weight(p: Dict[str, float]) -> float:
   return max((abs(float(w)) for w in p.values()), default=0.0)
