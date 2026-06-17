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

    def schedule(self, execution_time, ready_time):
        """
        Returns:
        - EST
        - CT
        """

        est = max(self.available_time, ready_time)
        ct = est + execution_time
        self.available_time = ct
        self.queue.append((est, ct))
        return est, ct


class ComputeNode:
    """
    VE or VES.
    """

    def __init__(self, node_id, compute_power, num_processors, x, y, node_type="VE"):
        self.node_id = node_id
        self.compute_power = compute_power
        self.num_processors = num_processors
        self.node_type = node_type
        self.x = x
        self.y = y
        self.processors = [Processor(i) for i in range(num_processors)]

    def get_lightest_processor(self):
        return min(self.processors, key=lambda p: p.available_time)

    def execution_time(self, cpu_cycles):
        return cpu_cycles / self.compute_power

    def schedule_task(self, cpu_cycles, ready_time):
        processor = self.get_lightest_processor()
        exec_time = self.execution_time(cpu_cycles)
        est, ct = processor.schedule(execution_time=exec_time, ready_time=ready_time)

        return {
            "processor_id": processor.processor_id,
            "EST": est,
            "CT": ct,
            "execution_time": exec_time,
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

    def __init__(self, dag, transmission_model, devices):
        self.dag = dag
        self.transmission_model = transmission_model
        self.devices = devices
        self.node_finish_times = {}
        self.node_schedule_info = {}

    def _edge_transfer_time(self, pred, node_id, pred_device, target_device):
        if pred == 0:
            edge_data = self.dag.nodes[node_id].get("data_size", 0.0)
        else:
            edge_data = self.dag.edges[pred, node_id].get("data_size", 0.0)

        same_location = pred_device.node_id == target_device.node_id
        distance = self.transmission_model.euclidean_distance(
            pred_device.x,
            pred_device.y,
            target_device.x,
            target_device.y,
        )

        return self.transmission_model.transmission_time(
            data_size_kb=edge_data,
            distance_m=distance,
            same_location=same_location,
        )

    # ==================================================
    # predecessor communication delay
    # ==================================================
    def predecessor_ready_time(self, node_id, target_device_id):
        predecessors = list(self.dag.predecessors(node_id))

        if len(predecessors) == 0:
            return 0.0

        ready_times = []
        target_device = self.devices[target_device_id]

        for pred in predecessors:
            if pred == 0:
                producer_vehicle_id = node_id[0]
                pred_device = self.devices[producer_vehicle_id]
                pred_ct = 0.0
            else:
                pred_info = self.node_schedule_info.get(pred)
                if pred_info is None:
                    raise ValueError(
                        f"Cannot schedule {node_id}: predecessor {pred} has not been scheduled."
                    )
                pred_device = self.devices[pred_info["device_id"]]
                pred_ct = pred_info["CT"]

            tx_time = self._edge_transfer_time(
                pred=pred,
                node_id=node_id,
                pred_device=pred_device,
                target_device=target_device,
            )
            # print("tx, pred ct")
            # print(tx_time,pred_ct)
            ready_times.append(pred_ct + tx_time)

        return max(ready_times) if ready_times else 0.0

    # ==================================================
    # equation (10) + (11)
    # ==================================================
    def schedule_node(self, node_id, device_id):
        node_attr = self.dag.nodes[node_id]
        cpu_cycles = node_attr["cpu_cycles"]
        device = self.devices[device_id]

        predecessor_ready = self.predecessor_ready_time(
            node_id=node_id,
            target_device_id=device_id,
        )

        result = device.schedule_task(
            cpu_cycles=cpu_cycles,
            ready_time=predecessor_ready,
        )
        # print(result)
        self.node_finish_times[node_id] = result["CT"]
        self.node_schedule_info[node_id] = {
            "device_id": device_id,
            "EST": result["EST"],
            "CT": result["CT"],
            "processor_id": result["processor_id"],
        }
        self.dag.nodes[node_id]["scheduled_location"] = device_id

        return result

    # ==================================================
    # node availability
    # ==================================================
    def update_available_nodes(self):
        for node in self.dag.nodes:
            if node == 0:
                continue

            if self.dag.nodes[node]["scheduled_location"] != -1:
                self.dag.nodes[node]["available"] = 0
                continue

            predecessors = list(self.dag.predecessors(node))
            ready = True

            for pred in predecessors:
                if pred == 0:
                    continue

                if self.dag.nodes[pred]["scheduled_location"] == -1:
                    ready = False
                    break

            self.dag.nodes[node]["available"] = int(ready)

    # ==================================================
    # completion check
    # ==================================================
    def is_all_scheduled(self):
        for node in self.dag.nodes:
            if node in [0]:
                continue

            if self.dag.nodes[node]["scheduled_location"] == -1:
                return False

        return True


def visualize_env(devices):
    import matplotlib.pyplot as plt

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
            label=node.node_type
            if node.node_type not in plt.gca().get_legend_handles_labels()[1]
            else "",
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

def main():
    devices = {
        0: ComputeNode(
            node_id=0,
            compute_power=1e9,
            num_processors=1,
            x=0,
            y=0,
            node_type="VE",
        ),
        1: ComputeNode(
            node_id=1,
            compute_power=1e9,
            num_processors=1,
            x=80,
            y=80,
            node_type="VE",
        ),
        2: ComputeNode(
            node_id=2,
            compute_power=10e9,
            num_processors=4,
            x=50,
            y=50,
            node_type="VES",
        ),
    }
    x = deepcopy(devices)
    return
if __name__ == "__main__":

    from dag_generator import DAGGenerator
    from transmission import TransmissionModel

    dag = DAGGenerator(num_tasks=2, num_nodes=10, max_out_degree=3).generate()

    transmission_model = TransmissionModel()
    
    devices = {
        0: ComputeNode(
            node_id=0,
            compute_power=1e9,
            num_processors=1,
            x=0,
            y=0,
            node_type="VE",
        ),
        1: ComputeNode(
            node_id=1,
            compute_power=1e9,
            num_processors=1,
            x=80,
            y=80,
            node_type="VE",
        ),
        2: ComputeNode(
            node_id=2,
            compute_power=10e9,
            num_processors=4,
            x=50,
            y=50,
            node_type="VES",
        ),
    }
    visualize_env(devices)
    scheduler = Scheduler(dag=dag, transmission_model=transmission_model, devices=devices)
    x = deepcopy(scheduler)
    available_nodes = [n for n in dag.nodes if dag.nodes[n]["available"] == 1]

    print("Available Nodes:", available_nodes)

    scheduler.update_available_nodes()

    available_nodes = [n for n in dag.nodes if dag.nodes[n]["available"] == 1]

    print("Available Nodes:", available_nodes)

    for node in available_nodes:
        if node == 0:
            continue

        result = scheduler.schedule_node(node_id=node, device_id=1)

        print(f"\nNode {node}")
        print(result)

    scheduler.update_available_nodes()
    available_nodes = [n for n in dag.nodes if dag.nodes[n]["available"] == 1]

    print("Available Nodes:", available_nodes)
