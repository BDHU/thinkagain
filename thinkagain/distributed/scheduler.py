"""Placement scheduler for distributing replica instances across nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nodes import NodeState


class PlacementScheduler:
    """Decides which nodes to place replica instances on."""

    def __init__(self, nodes: dict[str, "NodeState"]):
        self.nodes = nodes

    def select_nodes(
        self,
        replica_name: str,
        count: int,
        cpus_per_instance: int,
        gpus_per_instance: int,
    ) -> list[str]:
        if count == 0:
            return []

        placements: list[str] = []
        needs_gpu = gpus_per_instance > 0

        def key(node: "NodeState") -> tuple:
            has_gpu = node.config.gpus > 0
            gpu_priority = -1 if (needs_gpu and has_gpu) else 0
            return (gpu_priority, -node.available_cpus, -node.available_gpus)

        sorted_nodes = sorted(self.nodes.values(), key=key)

        for _ in range(count):
            placed = False
            for node in sorted_nodes:
                if node.can_fit(cpus_per_instance, gpus_per_instance):
                    placements.append(node.config.host)
                    node.reserve(cpus_per_instance, gpus_per_instance)
                    placed = True
                    break
            if not placed:
                self._revert_placements(
                    placements, cpus_per_instance, gpus_per_instance
                )
                raise RuntimeError(
                    f"Cannot place {count} instances of '{replica_name}' "
                    f"requiring {cpus_per_instance} CPUs and {gpus_per_instance} GPUs: "
                    f"insufficient cluster resources"
                )

        return placements

    def _revert_placements(
        self, placements: list[str], cpus_per_instance: int, gpus_per_instance: int
    ) -> None:
        for host in placements:
            self.nodes[host].release(cpus_per_instance, gpus_per_instance)

    def get_cluster_state(self) -> dict[str, dict]:
        return {
            host: {
                "total_cpus": state.config.cpus,
                "available_cpus": state.available_cpus,
                "total_gpus": state.config.gpus,
                "available_gpus": state.available_gpus,
                "servers": state.server_count,
                "utilization": state.utilization,
            }
            for host, state in self.nodes.items()
        }
