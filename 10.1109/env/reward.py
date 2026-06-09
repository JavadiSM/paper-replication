class RewardCalculator:

    def __init__(self):
        self.prev_cft = None
        self.local_baseline = None

    def reset(self):
        self.prev_cft = None
        self.local_baseline = None

    def compute_scaled_reward(self, scheduler, dag, devices):

        if self.local_baseline is None:
            self.local_baseline = scheduler.estimate_local_mean_cft()

        current_cft = scheduler.estimate_mean_cft()

        if self.prev_cft is None:
            reward = 0.0
        else:
            reward = (
                self.prev_cft - current_cft
            ) / max(self.local_baseline, 1e-9)

        self.prev_cft = current_cft

        completed_nodes = len(scheduler.node_schedule_info)
        total_nodes = len([n for n in dag.nodes if isinstance(n, tuple)])

        success_rate = completed_nodes / total_nodes if total_nodes > 0 else 0.0
        throughput = completed_nodes / current_cft if current_cft > 0 else 0.0

        return float(reward), {
            "reward": float(reward),
            "mean_cft": float(current_cft),
            "baseline_cft": float(self.local_baseline),
            "success_rate": float(success_rate),
            "throughput": float(throughput),
        }

    def final_reward(self, scheduler, dag, devices):
        return 0.0