from copy import deepcopy
import networkx as nx
import numpy as np


class RewardCalculator:

    def __init__(self):
        self.prev_cft = None
        self.local_baseline = None
        self.prev_scheduler_snapshot = None

    def reset(self):
        self.prev_cft = None
        self.local_baseline = None
        self.prev_scheduler_snapshot = None

    def _mean_cft(self, scheduler):

        task_finish_time = {}

        for node, info in scheduler.node_schedule_info.items():

            task_id = node[0]
            ct = info["CT"]

            if task_id not in task_finish_time:
                task_finish_time[task_id] = ct
            else:
                task_finish_time[task_id] = max(
                    task_finish_time[task_id],
                    ct
                )

        return float(np.mean(list(task_finish_time.values())))

    def _estimate_future_cft(self, scheduler):

        sim = deepcopy(scheduler)

        for node in nx.topological_sort(sim.dag):

            if node == 0:
                continue

            if not isinstance(node, tuple):
                continue

            if sim.dag.nodes[node]["scheduled_location"] != -1:
                continue

            sim.schedule_node(
                node_id=node,
                device_id=node[0]
            )

        return self._mean_cft(sim)

    def _estimate_local_baseline(self, scheduler):

        sim = deepcopy(scheduler)

        sim.node_finish_times = {}
        sim.node_schedule_info = {}

        for device in sim.devices.values():
            for proc in device.processors:
                proc.available_time = 0.0
                proc.queue.clear()

        for node in sim.dag.nodes:
            if node == 0:
                continue

            sim.dag.nodes[node]["scheduled_location"] = -1

        for node in nx.topological_sort(sim.dag):

            if node == 0:
                continue

            if not isinstance(node, tuple):
                continue

            sim.schedule_node(
                node_id=node,
                device_id=node[0]
            )

        return self._mean_cft(sim)

    # =========================================================
    # CORE FIX: paper-consistent reward
    # =========================================================
    def compute_scaled_reward(self, scheduler, dag, devices):

        # initialize baseline once
        if self.local_baseline is None:
            self.local_baseline = self._estimate_local_baseline(scheduler)

        # compute current full-system CFT after action t
        current_cft = self._estimate_future_cft(scheduler)

        # first step has no previous state
        if self.prev_cft is None:
            reward = self._estimate_local_baseline(scheduler) - current_cft
        else:
            # paper: reduction in CFT caused by action t
            reward = self.prev_cft - current_cft

        # update state
        self.prev_cft = current_cft

        # =====================================================
        # diagnostics (unchanged)
        # =====================================================
        completed_nodes = len(scheduler.node_schedule_info)
        total_nodes = len([n for n in dag.nodes if isinstance(n, tuple)])

        success_rate = (
            completed_nodes / total_nodes
            if total_nodes > 0 else 0.0
        )

        throughput = (
            completed_nodes / current_cft
            if current_cft > 0 else 0.0
        )

        return float(reward), {
            "reward": float(reward),
            "mean_cft": float(current_cft),
            "baseline_cft": float(self.local_baseline),
            "success_rate": float(success_rate),
            "throughput": float(throughput),
        }

    def final_reward(self, scheduler, dag, devices):
        return 0.0