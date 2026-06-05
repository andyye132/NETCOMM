from typing import List, Tuple

import numpy as np

from netcomm.types import Packet


class PoissonGenerator:
    def __init__(self, rates_per_priority: Tuple[float, ...], dt: float,
                 T_AoI: float = 0.1, seed: int = 0):
        self.rates = tuple(float(r) for r in rates_per_priority)
        self.dt = float(dt)
        self.T_AoI = float(T_AoI)
        self.rng = np.random.default_rng(seed)
        # Per-priority deadline scale; higher priority -> tighter deadline.
        self._deadline_scale = tuple(
            max(0.25, 1.0 - 0.2 * p) for p in range(len(self.rates))
        )

    def _generate_priority(self, t: float, dt: float,
                           src_dst_pairs: List[Tuple[int, int]],
                           next_id: int, priority: int) -> Tuple[List[Packet], int]:
        rate = self.rates[priority] if priority < len(self.rates) else 0.0
        if rate <= 0.0 or not src_dst_pairs:
            return [], next_id
        n_pairs = len(src_dst_pairs)
        # Expected packets per flow this step:
        lam = rate * dt
        out: List[Packet] = []
        for (s, d) in src_dst_pairs:
            n = int(self.rng.poisson(lam))
            for _ in range(n):
                deadline = t + self.T_AoI * self._deadline_scale[priority]
                out.append(Packet(
                    id=next_id, src=int(s), dst=int(d), t_gen=float(t),
                    deadline=float(deadline), priority=int(priority), size=1,
                ))
                next_id += 1
        return out, next_id

    def generate(self, t: float, dt: float,
                 src_dst_pairs: List[Tuple[int, int]],
                 next_id: int) -> Tuple[List[Packet], int]:
        all_pkts: List[Packet] = []
        for p in range(len(self.rates)):
            pkts, next_id = self._generate_priority(t, dt, src_dst_pairs, next_id, p)
            all_pkts.extend(pkts)
        return all_pkts, next_id


class BurstyGenerator(PoissonGenerator):
    def __init__(self, rates_per_priority, dt, T_AoI=0.1, seed=0,
                 burst_prob: float = 0.02, burst_size: int = 20,
                 burst_priority: int = 1):
        super().__init__(rates_per_priority, dt, T_AoI, seed)
        self.burst_prob = float(burst_prob)
        self.burst_size = int(burst_size)
        self.burst_priority = int(burst_priority)

    def generate(self, t, dt, src_dst_pairs, next_id):
        pkts, next_id = super().generate(t, dt, src_dst_pairs, next_id)
        # why: Test 9 UDP stress — occasional bursts on a random flow.
        if src_dst_pairs and self.rng.random() < self.burst_prob:
            s, d = src_dst_pairs[self.rng.integers(0, len(src_dst_pairs))]
            for _ in range(self.burst_size):
                pkts.append(Packet(
                    id=next_id, src=int(s), dst=int(d), t_gen=float(t),
                    deadline=float(t + self.T_AoI),
                    priority=self.burst_priority, size=1,
                ))
                next_id += 1
        return pkts, next_id
