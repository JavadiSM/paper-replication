import heapq
from collections import deque
import networkx as nx

import numpy as np
from copy import deepcopy

 
class Processor:
    """
    Single processor inside VE or VES.
    """
    def __init__(self, processor_id):
        self.processor_id = processor_id
        self.available_time = 0.0
        self.queue = deque()

    def schedule(
        self,
        execution_time,
        ready_time
    ):
        """
        Returns:
        - EST
        - CT
        """

        est = max(
            self.available_time,
            ready_time
        )
        ct = est + execution_time
        self.available_time = ct
        self.queue.append(
            (est, ct)
        )
        return est, ct


class ComputeNode:
    """
    VE or VES
    """

    def __init__(
        self,
        node_id,
        compute_power,
        num_processors,
        x,
        y,
        node_type="VE"
    ):

        self.node_id = node_id

        self.compute_power = compute_power

        self.num_processors = num_processors

        self.node_type = node_type

        self.x = x
        self.y = y

        self.processors = [
            Processor(i)
            for i in range(num_processors)
        ]

    def get_lightest_processor(self):

        return min(
            self.processors,
            key=lambda p: p.available_time
        )

    def execution_time(
        self,
        cpu_cycles
    ):

        return cpu_cycles / self.compute_power

    def schedule_task(
        self,
        cpu_cycles,
        ready_time
    ):

        processor = self.get_lightest_processor()

        exec_time = self.execution_time(cpu_cycles)

        est, ct = processor.schedule(
            execution_time=exec_time,
            ready_time=ready_time
        )

        return {
            "processor_id": processor.processor_id,
            "EST": est,
            "CT": ct,
            "execution_time": exec_time
        }


class Scheduler:
    """
    Implements:
    - EST
    - CT
    - FAT
    - FIFO execution
    - predecessor dependency logic
    """

    def __init__(self, dag, transmission_model,devices):

        self.dag = dag

        self.transmission_model = transmission_model

        self.devices = devices

        self.node_finish_times = {}

        self.node_schedule_info = {}

    # ==================================================
    # predecessor communication delay
    # ==================================================
    # ==================================================
    # predecessor communication delay
    # ==================================================

    def predecessor_ready_time(self,node_id,target_device_id):

        predecessors = list(
            self.dag.predecessors(node_id)
        )

        if len(predecessors) == 0:
            return 0.0

        ready_times = []

        target_device = self.devices[
            target_device_id
        ]

        for pred in predecessors:

            # =============================================
            # global start node
            #
            # semantics:
            # data originates from producer vehicle
            # =============================================

            if pred == 0:

                producer_vehicle_id = node_id[0]

                pred_device = self.devices[
                    producer_vehicle_id
                ]

                pred_ct = 0.0

                edge_data = self.dag.nodes[
                    node_id
                ].get(
                    "data_size",
                    0.0
                )

            # =============================================
            # normal predecessor
            # =============================================

            else:

                pred_info = self.node_schedule_info.get(
                    pred
                )

                # predecessor not finished yet
                if pred_info is None:
                    continue

                pred_device = self.devices[
                    pred_info["device_id"]
                ]

                pred_ct = pred_info["CT"]

                edge_data = self.dag.edges[
                    pred,
                    node_id
                ].get(
                    "data_size",
                    0.0
                )

            # =============================================
            # transmission
            # =============================================

            same_location = (

                    pred_device.node_id
                    ==
                    target_device.node_id
            )

            distance = self.transmission_model.euclidean_distance(

                pred_device.x,
                pred_device.y,

                target_device.x,
                target_device.y
            )

            tx_time = self.transmission_model.transmission_time(

                data_size_kb=edge_data,

                distance_m=distance,

                same_location=same_location
            )

            ready_times.append(
                pred_ct + tx_time
            )

        if len(ready_times) == 0:
            return 0.0

        return max(ready_times)
    # ==================================================
    # equation (10) + (11)
    # ==================================================

    def schedule_node(
        self,
        node_id,
        device_id
    ):

        node_attr = self.dag.nodes[node_id]

        cpu_cycles = node_attr["cpu_cycles"]

        device = self.devices[device_id]

        predecessor_ready = self.predecessor_ready_time(
            node_id=node_id,
            target_device_id=device_id
        )

        result = device.schedule_task(
            cpu_cycles=cpu_cycles,
            ready_time=predecessor_ready
        ) 
        """
                {
                    "processor_id": processor.processor_id,
                    "EST": est,
                    "CT": ct,
                    "execution_time": exec_time
                }
        """
        self.node_finish_times[node_id] = result["CT"]

        self.node_schedule_info[node_id] = {
            "device_id": device_id,
            "EST": result["EST"],
            "CT": result["CT"],
            "processor_id": result["processor_id"]
        }

        self.dag.nodes[node_id][
            "scheduled_location"
        ] = device_id

        return result

    # ==================================================
    # node availability
    # ==================================================

    def update_available_nodes(self):

        for node in self.dag.nodes:

            if node == 0:
                continue

            if self.dag.nodes[node][
                "scheduled_location"
            ] != -1:
                continue

            predecessors = list(
                self.dag.predecessors(node)
            )

            ready = True

            for pred in predecessors:

                # Start node is always ready
                if pred == 0:
                    continue

                if self.dag.nodes[pred][
                    "scheduled_location"
                ] == -1:
                    ready = False
                    break

            self.dag.nodes[node][
                "available"
            ] = int(ready)

    # ==================================================
    # completion check
    # ==================================================

    def is_all_scheduled(self):

        for node in self.dag.nodes:

            if node in [0]:
                continue

            if self.dag.nodes[node][
                "scheduled_location"
            ] == -1:

                return False

        return True




    def estimate_mean_cft(self):

        ct_cache = {}

        # ---------------------------------------
        # simulate processor availability
        # ---------------------------------------
        proc_available = {
            dev_id: [
                p.available_time for p in dev.processors
            ]
            for dev_id, dev in self.devices.items()
        }

        # ---------------------------------------
        # load already scheduled nodes
        # ---------------------------------------
        for node, info in self.node_schedule_info.items():
            ct_cache[node] = info["CT"]

            # reserve processor state (IMPORTANT)
            dev_id = info["device_id"]
            proc_idx = info["processor_id"]

            # sync simulated processor timeline
            proc_available[dev_id][proc_idx] = info["CT"]

        topo_order = list(nx.topological_sort(self.dag))

        # ---------------------------------------
        # simulate unscheduled nodes
        # ---------------------------------------
        for node in topo_order:

            if node == 0:
                continue

            if node in ct_cache:
                continue

            if not isinstance(node, tuple):
                continue

            device_id = node[0]
            device = self.devices[device_id]

            # -------------------------
            # compute ready_time
            # -------------------------
            ready_time = 0.0

            for pred in self.dag.predecessors(node):

                if pred == 0:
                    pred_ct = 0.0
                    pred_device = self.devices[node[0]]
                    edge_data = self.dag.nodes[node].get("data_size", 0.0)

                else:
                    pred_ct = ct_cache[pred]

                    if pred in self.node_schedule_info:
                        pred_device = self.devices[
                            self.node_schedule_info[pred]["device_id"]
                        ]
                    else:
                        pred_device = self.devices[pred[0]]

                    edge_data = self.dag.edges[pred, node].get("data_size", 0.0)

                same_location = (pred_device.node_id == device.node_id)

                distance = self.transmission_model.euclidean_distance(
                    pred_device.x,
                    pred_device.y,
                    device.x,
                    device.y
                )

                tx_time = self.transmission_model.transmission_time(
                    data_size_kb=edge_data,
                    distance_m=distance,
                    same_location=same_location
                )

                ready_time = max(ready_time, pred_ct + tx_time)

            exec_time = self.dag.nodes[node]["cpu_cycles"] / device.compute_power

            # ---------------------------------------
            # REAL scheduling (FIFO like Scheduler)
            # ---------------------------------------
            proc_list = proc_available[device_id]

            proc_idx = int(np.argmin(proc_list))

            est = max(ready_time, proc_list[proc_idx])
            ct = est + exec_time

            proc_list[proc_idx] = ct

            ct_cache[node] = ct

        # ---------------------------------------
        # end nodes CFT
        # ---------------------------------------
        tuple_nodes = [n for n in self.dag.nodes if isinstance(n, tuple)]

        end_id = max(n[1] for n in tuple_nodes)

        end_nodes = [n for n in tuple_nodes if n[1] == end_id]

        cfts = [ct_cache[n] for n in end_nodes]

        return float(np.mean(cfts))

    def estimate_local_mean_cft(self):

        ct_cache = {}

        topo_order = list(nx.topological_sort(self.dag))

        for node in topo_order:

            if node == 0:
                continue

            if not isinstance(node, tuple):
                continue

            device = self.devices[node[0]]

            predecessors = list(self.dag.predecessors(node))

            ready_time = 0.0

            for pred in predecessors:

                if pred == 0:
                    pred_ct = 0.0
                    pred_device = self.devices[node[0]]
                    edge_data = self.dag.nodes[node].get("data_size", 0.0)

                else:
                    pred_ct = ct_cache[pred]
                    pred_device = self.devices[pred[0]]
                    edge_data = self.dag.edges[pred, node].get("data_size", 0.0)

                tx_time = self.transmission_model.transmission_time(
                    data_size_kb=edge_data,
                    distance_m=0.0,
                    same_location=True
                )

                ready_time = max(
                    ready_time,
                    pred_ct + tx_time
                )

            exec_time = (
                self.dag.nodes[node]["cpu_cycles"]
                / device.compute_power
            )

            ct_cache[node] = ready_time + exec_time

        tuple_nodes = [n for n in self.dag.nodes if isinstance(n, tuple)]

        end_id = max(n[1] for n in tuple_nodes)

        end_nodes = [n for n in tuple_nodes if n[1] == end_id]

        cfts = [ct_cache[n] for n in end_nodes]

        return sum(cfts) / len(cfts)
import matplotlib.pyplot as plt
def visualize_env(devices):
    type_styles = {
        "VE": {"color": "blue", "marker": "o"},
        "VES": {"color": "red", "marker": "s"},
    }

    plt.figure(figsize=(8, 6))

    for device_id, node in devices.items():
        style = type_styles.get(node.node_type, {"color": "gray", "marker": "x"})

        plt.scatter(
            node.x,
            node.y,
            color=style["color"],
            marker=style["marker"],
            s=200,
            label=node.node_type if node.node_type not in plt.gca().get_legend_handles_labels()[1] else ""
        )

        label = (
            f"ID: {node.node_id}\n"
            f"Type: {node.node_type}\n"
            f"Power: {node.compute_power / 1e9:.1f} GFLOPS\n"
            f"Proc: {node.num_processors}"
        )
        plt.text(node.x + 3, node.y + 3, label, fontsize=9)

    plt.title("Compute Nodes Visualization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.axis("equal")
    plt.show()



if __name__ == "__main__":

    from dag_generator import DAGGenerator
    from transmission import TransmissionModel

    dag = DAGGenerator(
        num_tasks=2,
        num_nodes=10,
        max_out_degree=3
    ).generate()

    transmission_model = TransmissionModel()

    devices = {
        0: ComputeNode(
            node_id=0,
            compute_power=1e9,
            num_processors=1,
            x=0,
            y=0,
            node_type="VE"
        ),
        1: ComputeNode(
            node_id=1,
            compute_power=1e9,
            num_processors=1,
            x=80,
            y=80,
            node_type="VE"
        ),
        2: ComputeNode(
            node_id=2,
            compute_power=10e9,
            num_processors=4,
            x=50,
            y=50,
            node_type="VES"
        )
    }
    visualize_env(devices)
    scheduler = Scheduler(
        dag=dag,
        transmission_model=transmission_model,
        devices=devices
    )
    available_nodes = [
        n for n in dag.nodes
        if dag.nodes[n]["available"] == 1
    ]

    print("Available Nodes:", available_nodes)

    scheduler.update_available_nodes()

    available_nodes = [
        n for n in dag.nodes
        if dag.nodes[n]["available"] == 1
    ]

    print("Available Nodes:", available_nodes)

    for node in available_nodes:

        if node == 0:
            continue

        result = scheduler.schedule_node(
            node_id=node,
            device_id=1
        )

        print(f"\nNode {node}")
        print(result)

    scheduler.update_available_nodes()
    available_nodes = [
        n for n in dag.nodes
        if dag.nodes[n]["available"] == 1
    ]

    print("Available Nodes:", available_nodes)
