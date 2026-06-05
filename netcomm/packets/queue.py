from typing import List

from netcomm.types import Packet


class PriorityQueue:
    def __init__(self):
        self._store: List[Packet] = []

    def extend(self, packets):
        if packets:
            self._store.extend(packets)

    def __len__(self):
        return len(self._store)

    def drain_for_step(self) -> List[Packet]:
        # why: drain everything sorted by (-priority, deadline, t_gen) so urgent and
        # closest-to-expiring packets get planning quanta first. Caller decides which
        # ones to requeue if compute budget runs out.
        out = list(self._store)
        out.sort(key=lambda p: (-int(p.priority), float(p.deadline), float(p.t_gen)))
        self._store = []
        return out

    def requeue_unfinished(self, packets):
        if packets:
            self._store.extend(packets)
