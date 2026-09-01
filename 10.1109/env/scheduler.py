from copy import copy, deepcopy
from dataclasses import dataclass, field


@dataclass
class CPU:
    cpu_id: int
    timeline: list = field(default_factory=list)

    def earliest_slot(self, duration, ready_time):
        # First gap
        start = ready_time
        for item in self.timeline:
            if start + duration <= item["start"]:
                break
            start = max(start, item["finish"])
        return start, start + duration

    def reserve(self, node_id, start, finish):
        # CPU slot
        self.timeline.append({
            "node_id": node_id,
            "start": start,
            "finish": finish,
        })
        self.timeline.sort(key=lambda item: item["start"])

    def reset(self):
        self.timeline.clear()


@dataclass
class VM:
    vm_id: int
    speed: float
    num_cpus: int = 1
    active_power_w: float = 0.0
    cpus: list = field(init=False)

    def __post_init__(self):
        self.cpus = [CPU(i) for i in range(self.num_cpus)]

    def execution_time(self, cpu_cycles):
        # Compute time
        return cpu_cycles / self.speed

    def schedule(self, node_id, cpu_cycles, ready_time):
        # CPU choice
        duration = self.execution_time(cpu_cycles)
        choices = []

        for cpu in self.cpus:
            start, finish = cpu.earliest_slot(duration, ready_time)
            choices.append((finish, start, cpu.cpu_id, cpu))

        finish, start, cpu_id, cpu = min(choices)
        cpu.reserve(node_id, start, finish)

        return {
            "cpu_id": cpu_id,
            "EST": start,
            "CT": finish,
            "execution_time": duration,
            "compute_energy_j": self.active_power_w * duration,
        }

    def reset(self):
        for cpu in self.cpus:
            cpu.reset()


@dataclass
class Channel:
    channel_id: int
    timeline: list = field(default_factory=list)

    def reserve(self, edge, start, finish):
        # Channel slot
        self.timeline.append({
            "edge": edge,
            "start": start,
            "finish": finish,
        })
        self.timeline.sort(key=lambda item: item["start"])

    def reset(self):
        self.timeline.clear()


class ComputeDevice:
    def __init__(
        self,
        device_id,
        device_type,
        vms,
        x=0.0,
        y=0.0,
        tx_channels=1,
        rx_channels=1,
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.x = x
        self.y = y
        self.vms = {vm.vm_id: vm for vm in vms}
        self.tx_channels = [Channel(i) for i in range(tx_channels)]
        self.rx_channels = [Channel(i) for i in range(rx_channels)]

    def reset(self):
        for vm in self.vms.values():
            vm.reset()

        for channel in self.tx_channels + self.rx_channels:
            channel.reset()


class Scheduler:
    def __init__(
        self,
        dag,
        transmission_model,
        devices,
        mobility=None,
        position_mode="offline",
        criticality_mode="LO",
    ):
        self.dag = dag
        self.transmission_model = transmission_model
        self.devices = {
            device.device_id: device
            for device in devices
        } if not isinstance(devices, dict) else devices

        self.mobility = mobility
        self.position_mode = position_mode
        self.criticality_mode = criticality_mode

        self.node_schedule_info = {}
        self.edge_schedule_info = {}

        self.compute_energy_j = 0.0
        self.communication_energy_j = 0.0

    def _runtime_copy(self):
        # Preview copy
        scheduler = copy(self)
        scheduler.dag = deepcopy(self.dag)
        scheduler.devices = deepcopy(self.devices)
        scheduler.node_schedule_info = deepcopy(
            self.node_schedule_info
        )
        scheduler.edge_schedule_info = deepcopy(
            self.edge_schedule_info
        )
        return scheduler

    def _position_at(self, device_id, time_s):
        # Device position
        device = self.devices[device_id]

        if self.position_mode == "offline":
            return device.x, device.y

        if callable(self.mobility):
            return self.mobility(device_id, time_s)

        return self.mobility.position_at(device_id, time_s)

    def _owner(self, node_id):
        # Task owner
        attr = self.dag.nodes[node_id]
        return attr.get(
            "owner_vehicle_id",
            attr.get("task_id"),
        )

    def _cpu_cycles(self, node_id):
        # Criticality load
        attr = self.dag.nodes[node_id]

        if (
            self.criticality_mode == "HI"
            and attr.get("criticality", 0) == 1
        ):
            return attr.get(
                "wcet_hi",
                attr.get("cpu_cycles", 0.0),
            )

        return attr.get(
            "wcet_lo",
            attr.get("cpu_cycles", 0.0),
        )

    def _edge_data(self, pred, node_id):
        # Edge payload
        edge = self.dag.edges[pred, node_id]
        return edge.get(
            "data_size_kb",
            edge.get("data_size", 0.0),
        )

    def _pred_source(self, pred, node_id):
        # Source state
        start_node = self.dag.graph.get("start_node", 0)

        if pred == start_node:
            task_id = self.dag.nodes[node_id]["task_id"]
            release_times = self.dag.graph.get(
                "release_times",
                {},
            )
            return (
                self._owner(node_id),
                release_times.get(task_id, 0.0),
            )

        info = self.node_schedule_info[pred]
        return info["device_id"], info["CT"]

    @staticmethod
    def _common_slot(
        tx_channel,
        rx_channel,
        duration,
        ready_time,
    ):
        # Common gap
        start = ready_time
        intervals = sorted(
            tx_channel.timeline + rx_channel.timeline,
            key=lambda item: item["start"],
        )

        for item in intervals:
            if start + duration <= item["start"]:
                break
            start = max(start, item["finish"])

        return start, start + duration

    def _channel_choice(
        self,
        source_id,
        target_id,
        duration,
        ready_time,
    ):
        # Channel choice
        choices = []
        source = self.devices[source_id]
        target = self.devices[target_id]

        for tx in source.tx_channels:
            for rx in target.rx_channels:
                start, finish = self._common_slot(
                    tx,
                    rx,
                    duration,
                    ready_time,
                )
                choices.append((
                    finish,
                    start,
                    tx.channel_id,
                    rx.channel_id,
                    tx,
                    rx,
                ))

        return min(choices)

    def _transfer(
        self,
        pred,
        node_id,
        target_device_id,
    ):
        # Edge transfer
        source_device_id, ready_time = self._pred_source(
            pred,
            node_id,
        )
        data_size_kb = self._edge_data(pred, node_id)

        source_xy = self._position_at(
            source_device_id,
            ready_time,
        )
        target_xy = self._position_at(
            target_device_id,
            ready_time,
        )

        distance_m = (
            self.transmission_model.euclidean_distance(
                *source_xy,
                *target_xy,
            )
        )

        if (
            source_device_id == target_device_id
            or distance_m == 0
            or data_size_kb == 0
        ):
            record = {
                "source_device_id": source_device_id,
                "target_device_id": target_device_id,
                "tx_channel_id": None,
                "rx_channel_id": None,
                "ready_time": ready_time,
                "send_start": ready_time,
                "arrival_time": ready_time,
                "distance_m": distance_m,
                "rate_bps": 0.0,
                "time_s": 0.0,
                "energy_j": 0.0,
            }
            self.edge_schedule_info[
                (pred, node_id)
            ] = record
            return record

        metrics = self.transmission_model.metrics(
            data_size_kb,
            distance_m,
        )

        (
            finish,
            start,
            tx_id,
            rx_id,
            tx,
            rx,
        ) = self._channel_choice(
            source_device_id,
            target_device_id,
            metrics["time_s"],
            ready_time,
        )

        tx.reserve((pred, node_id), start, finish)
        rx.reserve((pred, node_id), start, finish)

        record = {
            "source_device_id": source_device_id,
            "target_device_id": target_device_id,
            "tx_channel_id": tx_id,
            "rx_channel_id": rx_id,
            "ready_time": ready_time,
            "send_start": start,
            "arrival_time": finish,
            "distance_m": distance_m,
            "rate_bps": metrics["rate_bps"],
            "time_s": metrics["time_s"],
            "energy_j": metrics["energy_j"],
        }

        self.edge_schedule_info[
            (pred, node_id)
        ] = record

        self.communication_energy_j += metrics[
            "energy_j"
        ]

        return record

    def predecessor_ready_time(
        self,
        node_id,
        target_device_id,
    ):
        # Input arrivals
        predecessors = list(
            self.dag.predecessors(node_id)
        )

        predecessors.sort(
            key=lambda pred: self._pred_source(
                pred,
                node_id,
            )[1]
        )

        arrivals = [
            self._transfer(
                pred,
                node_id,
                target_device_id,
            )["arrival_time"]
            for pred in predecessors
        ]

        return max(arrivals, default=0.0)

    def schedule_node(
        self,
        node_id,
        device_id,
        vm_id=0,
    ):
        # Node schedule
        if self.dag.nodes[node_id].get(
            "node_type"
        ) == 2:
            return self.schedule_end(node_id)

        ready_time = self.predecessor_ready_time(
            node_id,
            device_id,
        )

        vm = self.devices[device_id].vms[vm_id]

        result = vm.schedule(
            node_id,
            self._cpu_cycles(node_id),
            ready_time,
        )

        result.update({
            "device_id": device_id,
            "vm_id": vm_id,
        })

        self.node_schedule_info[node_id] = result
        self.compute_energy_j += result[
            "compute_energy_j"
        ]

        self.dag.nodes[node_id][
            "scheduled_location"
        ] = device_id

        return result

    def schedule_end(self, node_id):
        # Task completion
        owner_id = self._owner(node_id)

        ready_time = self.predecessor_ready_time(
            node_id,
            owner_id,
        )

        result = {
            "device_id": owner_id,
            "vm_id": None,
            "cpu_id": None,
            "EST": ready_time,
            "CT": ready_time,
            "execution_time": 0.0,
            "compute_energy_j": 0.0,
        }

        self.node_schedule_info[node_id] = result

        self.dag.nodes[node_id][
            "scheduled_location"
        ] = owner_id

        return result

    def preview_node(
        self,
        node_id,
        device_id,
        vm_id=0,
    ):
        # Candidate result
        scheduler = self._runtime_copy()

        result = scheduler.schedule_node(
            node_id,
            device_id,
            vm_id,
        )

        result["communication_energy_j"] = (
            scheduler.communication_energy_j
            - self.communication_energy_j
        )

        return result

    def update_available_nodes(self):
        # DAG readiness
        start_node = self.dag.graph.get(
            "start_node",
            0,
        )

        for node_id in self.dag.nodes:
            if node_id == start_node:
                continue

            scheduled = (
                node_id in self.node_schedule_info
            )

            predecessors = [
                pred
                for pred in self.dag.predecessors(
                    node_id
                )
                if pred != start_node
            ]

            ready = all(
                pred in self.node_schedule_info
                for pred in predecessors
            )

            self.dag.nodes[node_id][
                "available"
            ] = int(ready and not scheduled)

    def is_all_scheduled(self):
        # Schedule status
        start_node = self.dag.graph.get(
            "start_node",
            0,
        )

        return all(
            node_id == start_node
            or node_id in self.node_schedule_info
            for node_id in self.dag.nodes
        )

    def reset(self):
        # Runtime reset
        self.node_schedule_info.clear()
        self.edge_schedule_info.clear()

        self.compute_energy_j = 0.0
        self.communication_energy_j = 0.0

        for device in self.devices.values():
            device.reset()

        start_node = self.dag.graph.get(
            "start_node",
            0,
        )

        for node_id in self.dag.nodes:
            self.dag.nodes[node_id][
                "scheduled_location"
            ] = -1

            self.dag.nodes[node_id][
                "available"
            ] = int(node_id == start_node)

        self.update_available_nodes()

    def result(self):
        # Schedule metrics
        end_nodes = self.dag.graph.get(
            "end_nodes",
            {},
        )

        task_cft = {
            task_id: self.node_schedule_info[
                end_id
            ]["CT"]
            for task_id, end_id in end_nodes.items()
            if end_id in self.node_schedule_info
        }

        makespan = max(
            (
                info["CT"]
                for info
                in self.node_schedule_info.values()
            ),
            default=0.0,
        )

        mean_cft = (
            sum(task_cft.values()) / len(task_cft)
            if task_cft
            else 0.0
        )

        return {
            "makespan": makespan,
            "mean_cft": mean_cft,
            "task_cft": task_cft,
            "compute_energy_j": self.compute_energy_j,
            "communication_energy_j":
                self.communication_energy_j,
            "total_energy_j":
                self.compute_energy_j
                + self.communication_energy_j,
        }