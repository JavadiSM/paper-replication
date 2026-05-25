class RewardCalculator:
    """
    Implements equation (28) from the paper.

    Multi-task version.

    Node semantics:

        0
            global start node

        (task_id, node_id)
            actual task node

    task_id:
        producer vehicle id

    node_id:
        DAG node index

    if node_id == max_nodes:
        task finished
    """

    def __init__(self, local_device_id=0):

        self.local_device_id = local_device_id

    # ==================================================
    # executable node check
    # ==================================================

    @staticmethod
    def is_executable_node(node):

        return (
            isinstance(node, tuple)
            and len(node) == 2
        )

    # ==================================================
    # completion node check
    # ==================================================

    @staticmethod
    def is_terminal_node(node, dag):

        if not isinstance(node, tuple):
            return False

        max_node_id = max(
            n[1]
            for n in dag.nodes
            if isinstance(n, tuple)
        )

        return node[1] == max_node_id

    # ==================================================
    # full graph completion time
    # ==================================================

    def calculate_makespan(self, scheduler):
        """
        Overall completion time.

        Uses maximum CT among all scheduled nodes.
        """

        if (
            hasattr(scheduler, "node_finish_times")
            and len(scheduler.node_finish_times) > 0
        ):

            return max(
                scheduler.node_finish_times.values()
            )

        if (
            hasattr(scheduler, "node_schedule_info")
            and len(scheduler.node_schedule_info) > 0
        ):

            return max(
                info["CT"]
                for info in scheduler.node_schedule_info.values()
            )

        return 0.0

    # ==================================================
    # local-only estimation
    # ==================================================

    def estimate_local_completion(
        self,
        dag,
        devices
    ):
        """
        Baseline:
            execute each task on its producer vehicle.

        The total system completion time is the slowest producer vehicle,
        because vehicles execute their own DAGs in parallel.
        """
        completion_by_vehicle = {}

        executable_nodes = [

            n for n in dag.nodes

            if (
                self.is_executable_node(n)
                and not self.is_terminal_node(
                    n,
                    dag
                )
            )
        ]

        for node in executable_nodes:
            producer_vehicle_id = node[0]
            device = devices.get(
                producer_vehicle_id,
                devices[self.local_device_id]
            )

            cpu_cycles = dag.nodes[node].get(
                "cpu_cycles",
                0.0
            )

            exec_time = (
                cpu_cycles
                /
                device.compute_power
            )

            completion_by_vehicle[producer_vehicle_id] = (
                completion_by_vehicle.get(
                    producer_vehicle_id,
                    0.0
                )
                +
                exec_time
            )

        return max(
            completion_by_vehicle.values(),
            default=0.0
        )

    # ==================================================
    # incremental reward
    # ==================================================

    def compute_reward(
        self,
        previous_makespan,
        current_makespan
    ):
        """
        Positive:
            lower completion time

        Negative:
            higher completion time
        """

        reward = (
            previous_makespan
            -
            current_makespan
        )

        return float(reward)

    def compute_scaled_reward(
        self,
        previous_makespan,
        current_makespan,
        local_baseline
    ):
        """
        Incremental reward scaled by the local-only baseline.
        """

        reward = self.compute_reward(
            previous_makespan,
            current_makespan
        )

        if local_baseline <= 0:
            return reward

        return float(
            reward
            /
            local_baseline
        )

    # ==================================================
    # normalized reward
    # ==================================================

    def compute_normalized_reward(
        self,
        current_makespan,
        local_baseline
    ):
        """
        Compare against local-only execution.
        """

        if local_baseline <= 0:
            return 0.0

        reward = (
            local_baseline
            -
            current_makespan
        ) / local_baseline

        return float(reward)

    # ==================================================
    # terminal reward
    # ==================================================

    def final_reward(
        self,
        scheduler,
        dag,
        devices
    ):

        makespan = self.calculate_makespan(
            scheduler
        )

        baseline = self.estimate_local_completion(
            dag,
            devices
        )

        return self.compute_normalized_reward(
            current_makespan=makespan,
            local_baseline=baseline
        )


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    from dag_generator import DAGGenerator
    from transmission import TransmissionModel
    from scheduler import Scheduler, ComputeNode

    # --------------------------------------------------
    # generate multi-task DAG
    # --------------------------------------------------

    dag = DAGGenerator(
        num_nodes=10,
        max_out_degree=3,
        num_tasks=2
    ).generate()

    transmission_model = TransmissionModel()

    # --------------------------------------------------
    # devices
    # --------------------------------------------------

    devices = {

        # local vehicle
        0: ComputeNode(
            node_id=0,
            compute_power=1e9,
            num_processors=1,
            x=0,
            y=0,
            node_type="VE"
        ),

        # edge server
        1: ComputeNode(
            node_id=1,
            compute_power=10e9,
            num_processors=4,
            x=100,
            y=100,
            node_type="VES"
        )
    }

    # --------------------------------------------------
    # scheduler
    # --------------------------------------------------

    scheduler = Scheduler(
        dag=dag,
        transmission_model=transmission_model,
        devices=devices
    )

    scheduler.update_available_nodes()

    reward_calculator = RewardCalculator()

    previous_makespan = 0.0

    # --------------------------------------------------
    # available executable nodes
    # --------------------------------------------------

    available_nodes = [

        n for n in dag.nodes

        if (
            n != 0
            and isinstance(n, tuple)
            and dag.nodes[n]["available"] == 1
        )
    ]

    print("Available Nodes:")
    print(available_nodes)

    # --------------------------------------------------
    # scheduling loop
    # --------------------------------------------------

    for node in available_nodes:

        # terminal nodes are ignored
        if reward_calculator.is_terminal_node(
            node,
            dag
        ):
            continue

        # ----------------------------------------------
        # IMPORTANT:
        #
        # node[0] == producer vehicle
        #
        # execution device is selected independently
        # ----------------------------------------------

        scheduler.schedule_node(
            node_id=node,
            device_id=1
        )

        current_makespan = (
            reward_calculator.calculate_makespan(
                scheduler
            )
        )

        reward = reward_calculator.compute_reward(
            previous_makespan,
            current_makespan
        )

        previous_makespan = current_makespan

        print(
            f"Node {node} Reward: {reward:.4f}"
        )

    # --------------------------------------------------
    # final reward
    # --------------------------------------------------

    final_reward = reward_calculator.final_reward(
        scheduler,
        dag,
        devices
    )

    print("\nFinal Reward:")
    print(final_reward)
