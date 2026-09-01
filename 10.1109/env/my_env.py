from collections import deque, defaultdict
import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dag_generator import DAGGenerator
from env.reward import RewardCalculator
from env.scheduler import ComputeNode, Scheduler
from env.transmission import TransmissionModel

try:
    from env.mobility_replay import MobilityReplay
except Exception:
    MobilityReplay = None


def node_sort_key(node):
    if node == 0:
        return (0, -1, -1)

    if isinstance(node, tuple):
        task_id, node_id = node
        return (1, task_id, node_id)

    return (2, node, node)


class MobilityDataset:
    def __init__(self, dataset_path="simulation_results1.txt"):
        self.dataset_path = dataset_path
        self.traces = self._load_dataset()

    def _load_dataset(self):
        traces = defaultdict(list)

        with open(self.dataset_path, "r") as f:
            next(f, None)

            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                t, vid, x, y, speed, angle = parts
                traces[vid].append(
                    (
                        float(t),
                        float(x),
                        float(y),
                        float(speed),
                        float(angle),
                    )
                )

        for vid in traces:
            traces[vid].sort(key=lambda r: r[0])

        return dict(traces)

    def assign_vehicle_ids(self, num_vehicles, rng):
        trace_ids = list(self.traces.keys())
        if len(trace_ids) == 0:
            raise RuntimeError("No vehicle traces found in mobility dataset.")

        rng.shuffle(trace_ids)

        mapping = {}
        available = trace_ids.copy()

        for vehicle_id in range(num_vehicles):
            if not available:
                available = trace_ids.copy()

            chosen = rng.choice(available)
            available.remove(chosen)
            mapping[vehicle_id] = chosen

        return mapping


class VehicleMobility:
    """
    Legacy mobility replay. Kept for backward compatibility.
    """

    def __init__(self, trace, device_type="VE"):
        self.device_type = device_type
        self.trace = trace
        self.trace_index = 0

        self.x = 0.0
        self.y = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.heading = 0.0
        self.last_time = None

        if self.device_type == "VES":
            return

        if len(self.trace) > 0:
            self._apply_state(self.trace[0])

    def _apply_state(self, record):
        t, x, y, speed, angle = record

        prev_velocity = self.velocity

        if self.last_time is None:
            self.acceleration = 0.0
        else:
            dt = max(t - self.last_time, 1e-6)
            self.acceleration = (speed - prev_velocity) / dt

        self.last_time = t
        self.x = x
        self.y = y
        self.velocity = speed
        self.heading = np.deg2rad(angle)

    def step(self, dt=1.0):
        if self.device_type == "VES":
            return

        if not self.trace:
            return

        if self.trace_index + 1 >= len(self.trace):
            return

        self.trace_index += 1
        self._apply_state(self.trace[self.trace_index])


try:
    from env.dataload import *
except Exception:
    from dataload import *


class VEC(gym.Env):
    def __init__(
        self,
        num_ve: int = 5,
        num_ves: int = 1,
        num_nodes: int = 5,
        history_len: int = 4,
        use_time_dependent_mobility: bool = False,
        mobility_time_step: float = 1.0,
        mobility_dataset_path: str = "simulation_results1.txt",
    ):
        super(VEC, self).__init__()

        self.num_ve = num_ve
        self.num_ves = num_ves
        self.num_nodes = num_nodes
        self.history_len = history_len
        self.world_size = 10

        self.use_time_dependent_mobility = bool(use_time_dependent_mobility)
        self.mobility_time_step = float(mobility_time_step)
        self.mobility_dataset_path = mobility_dataset_path
        self.current_time = 0.0

        ((self.xmin, self.ymin), (self.xmax, self.ymax)) = (
            (290.62, 313.76),
            (4215.21, 3300.04),
        )

        self.mobility_dataset = MobilityDataset(dataset_path=mobility_dataset_path)
        self.mobility_replay = None
        if self.use_time_dependent_mobility:
            if MobilityReplay is None:
                raise ImportError(
                    "MobilityReplay is not available, but use_time_dependent_mobility=True."
                )
            self.mobility_replay = MobilityReplay(dataset_path=mobility_dataset_path)

        self.num_locations = self.num_ve + self.num_ves
        self._num_task_graph_nodes = num_ve * (num_nodes + 1) + 1

        self._python_rng = random.Random()
        self.dag_generator = DAGGenerator(
            num_tasks=num_ve,
            num_nodes=num_nodes,
            max_out_degree=(5 * num_nodes // 6),
            rng=self._python_rng,
        )

        self.reward_calculator = RewardCalculator()
        self.previous_makespan = 0.0
        self.transmission_model = TransmissionModel()

        self.devices = {}
        self.mobility_models = {}
        self.vehicle_trace_mapping = {}
        self.dag = self.dag_generator.generate()

        self.node_feature_dim = 6
        self.total_actions = self._num_task_graph_nodes * self.num_locations

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._num_task_graph_nodes, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "adj_matrix": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self._num_task_graph_nodes, self._num_task_graph_nodes),
                    dtype=np.float32,
                ),
                "trajectory": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_ve, self.history_len, 5),
                    dtype=np.float32,
                ),
                "locations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_locations, self.history_len, 6),
                    dtype=np.float32,
                ),
                "node_runtime": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_locations, 2),
                    dtype=np.float32,
                ),
            }
        )

        self.trajectory_buffers = {
            vehicle_id: deque(maxlen=self.history_len)
            for vehicle_id in range(self.num_ve)
        }
        self.location_buffers = {}
        self.scheduler = None

        self.reset()

    def _device_id(self, device_or_id):
        return device_or_id.node_id if hasattr(device_or_id, "node_id") else device_or_id

    def _build_devices(self):
        self.devices = {}
        self.mobility_models = {}
        self.vehicle_trace_mapping = {}

        if self.use_time_dependent_mobility:
            for vehicle_id in range(self.num_ve):
                self.devices[vehicle_id] = ComputeNode(
                    node_id=vehicle_id,
                    compute_power=1e9,
                    num_processors=1,
                    x=0.0,
                    y=0.0,
                    node_type="VE",
                )

            for ves_id in range(self.num_ves):
                device_id = self.num_ve + ves_id
                x = float(np.random.uniform(self.xmin, self.xmax))
                y = float(np.random.uniform(self.ymin, self.ymax))

                self.devices[device_id] = ComputeNode(
                    node_id=device_id,
                    compute_power=10e9,
                    num_processors=4,
                    x=x,
                    y=y,
                    node_type="VES",
                )

            if self.mobility_replay is not None:
                self.vehicle_trace_mapping = self.mobility_replay.bind_devices(
                    self.devices,
                    rng=self._python_rng,
                    time_ms=self.current_time,
                )

            self._sync_mobility_to_time(self.current_time)

        else:
            vehicle_mapping = self.mobility_dataset.assign_vehicle_ids(
                self.num_ve,
                self._python_rng,
            )
            self.vehicle_trace_mapping = vehicle_mapping

            for vehicle_id in range(self.num_ve):
                trace_id = vehicle_mapping[vehicle_id]
                trace = self.mobility_dataset.traces[trace_id]

                mobility = VehicleMobility(
                    trace=trace,
                    device_type="VE",
                )
                self.mobility_models[vehicle_id] = mobility

                self.devices[vehicle_id] = ComputeNode(
                    node_id=vehicle_id,
                    compute_power=1e9,
                    num_processors=1,
                    x=mobility.x,
                    y=mobility.y,
                    node_type="VE",
                )

            for ves_id in range(self.num_ves):
                device_id = self.num_ve + ves_id
                x = float(np.random.uniform(self.xmin, self.xmax))
                y = float(np.random.uniform(self.ymin, self.ymax))

                self.devices[device_id] = ComputeNode(
                    node_id=device_id,
                    compute_power=10e9,
                    num_processors=4,
                    x=x,
                    y=y,
                    node_type="VES",
                )

                ves_mobility = VehicleMobility(
                    trace=[],
                    device_type="VES",
                )
                ves_mobility.x = x
                ves_mobility.y = y
                self.mobility_models[device_id] = ves_mobility

        self.num_locations = len(self.devices)
        self.location_buffers = {
            device_id: deque(maxlen=self.history_len)
            for device_id in self.devices
        }

    def _get_vehicle_state_row(self, vehicle_id):
        device = self.devices[vehicle_id]

        if self.use_time_dependent_mobility:
            if self.mobility_replay is None:
                return [device.x, device.y, 0.0, 0.0, 0.0]

            assignment = self.mobility_replay.mapping.get(vehicle_id)
            if assignment is None:
                return [device.x, device.y, 0.0, 0.0, 0.0]

            state = self.mobility_replay.get_state(assignment.trace_id, self.current_time)
            return [state.x, state.y, state.speed, state.acceleration, state.heading_rad]

        mobility = self.mobility_models[vehicle_id]
        return [mobility.x, mobility.y, mobility.velocity, mobility.acceleration, mobility.heading]

    def _sync_mobility_to_time(self, time_ms):
        self.current_time = float(time_ms)

        if self.use_time_dependent_mobility:
            if self.mobility_replay is not None:
                self.mobility_replay.refresh_mapping(self.current_time, self._python_rng)

            for vehicle_id in range(self.num_ve):
                device = self.devices[vehicle_id]

                if self.mobility_replay is None:
                    continue

                assignment = self.mobility_replay.mapping.get(vehicle_id)
                if assignment is None:
                    continue

                state = self.mobility_replay.get_state(assignment.trace_id, self.current_time)
                device.x = state.x
                device.y = state.y

                row = [state.x, state.y, state.speed, state.acceleration, state.heading_rad]
                self.trajectory_buffers[vehicle_id].append(row)
        else:
            for vehicle_id, mobility in self.mobility_models.items():
                if self.devices[vehicle_id].node_type == "VES":
                    continue

                mobility.step()
                self.devices[vehicle_id].x = mobility.x
                self.devices[vehicle_id].y = mobility.y

                row = [mobility.x, mobility.y, mobility.velocity, mobility.acceleration, mobility.heading]
                self.trajectory_buffers[vehicle_id].append(row)

    def _sync_location_buffers(self):
        for device_id, device in self.devices.items():
            task_count = sum(
                1
                for info in self.scheduler.node_schedule_info.values()
                if info["device_id"] == device_id
            )

            current_finish_time = max(
                (processor.available_time for processor in device.processors),
                default=0.0,
            )

            state = [
                1.0,
                task_count,
                device.x,
                device.y,
                device.compute_power,
                current_finish_time,
            ]
            self.location_buffers[device_id].append(state)

    def _get_valid_nodes(self):
        valid_nodes = []
        for node in self.dag.nodes:
            if self._is_schedulable(node):
                valid_nodes.append(node)

        valid_nodes = sorted(valid_nodes, key=node_sort_key)
        return valid_nodes

    def _map_continuous_to_node(self, value):
        valid_nodes = self._get_valid_nodes()
        if len(valid_nodes) == 0:
            return None

        value = np.clip(value, -1.0, 1.0)
        scaled = (value + 1.0) / 2.0
        idx = scaled * (len(valid_nodes) - 1)
        idx = int(np.rint(idx))
        idx = np.clip(idx, 0, len(valid_nodes) - 1)
        return valid_nodes[idx]

    def _map_continuous_to_location(self, value):
        device_ids = sorted(list(self.devices.keys()))

        value = np.clip(value, -1.0, 1.0)
        scaled = (value + 1.0) / 2.0
        idx = scaled * (len(device_ids) - 1)
        idx = int(np.rint(idx))
        idx = np.clip(idx, 0, len(device_ids) - 1)
        return device_ids[idx]

    def _is_schedulable(self, node):
        if node == 0:
            return False

        attr = self.dag.nodes[node]
        return attr["available"] == 1 and attr["scheduled_location"] == -1

    def _get_info(self):
        return {
            "valid_nodes": self._get_valid_nodes(),
            "mean_cft": self.reward_calculator.prev_cft
            if hasattr(self.reward_calculator, "prev_cft")
            else None,
            "baseline_cft": self.reward_calculator.local_baseline
            if hasattr(self.reward_calculator, "local_baseline")
            else None,
        }

    def _init_buffers(self):
        self.trajectory_buffers = {
            vehicle_id: deque(maxlen=self.history_len)
            for vehicle_id in range(self.num_ve)
        }
        self.location_buffers = {
            device_id: deque(maxlen=self.history_len)
            for device_id in self.devices
        }

        for vehicle_id in range(self.num_ve):
            state = self._get_vehicle_state_row(vehicle_id)
            for _ in range(self.history_len):
                self.trajectory_buffers[vehicle_id].append(state)

        for device_id, device in self.devices.items():
            task_count = 0
            current_finish_time = max(
                (processor.available_time for processor in device.processors),
                default=0.0,
            )

            state = [
                1.0,
                task_count,
                device.x,
                device.y,
                device.compute_power,
                current_finish_time,
            ]

            for _ in range(self.history_len):
                self.location_buffers[device_id].append(state)

    def _get_observation(self):
        ordered_nodes = sorted(self.dag.nodes, key=node_sort_key)

        node_features = np.array(
            [
                [
                    self.dag.nodes[node]["cpu_cycles"],
                    self.dag.nodes[node]["data_size"],
                    self.dag.nodes[node]["in_degree"],
                    self.dag.nodes[node]["out_degree"],
                    self.dag.nodes[node]["scheduled_location"],
                    self.dag.nodes[node]["available"],
                ]
                for node in ordered_nodes
            ],
            dtype=np.float32,
        )

        adj = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=np.float32)
        node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
        for source, target in self.dag.edges:
            adj[node_to_idx[source], node_to_idx[target]] = 1.0

        trajectory = np.array(
            [list(self.trajectory_buffers[vehicle_id]) for vehicle_id in range(self.num_ve)],
            dtype=np.float32,
        )

        node_runtime = []
        for device_id, device in self.devices.items():
            current_finish_time = max(
                (processor.available_time for processor in device.processors),
                default=0.0,
            )
            node_runtime.append([device.compute_power, current_finish_time])

        locations = np.array(
            [list(self.location_buffers[device_id]) for device_id in self.devices],
            dtype=np.float32,
        )

        return {
            "node_features": node_features,
            "adj_matrix": adj,
            "trajectory": trajectory,
            "locations": np.array(locations, dtype=np.float32),
            "node_runtime": np.array(node_runtime, dtype=np.float32),
        }

    def reset(self, seed=None):  # type: ignore
        super().reset(seed=seed)

        if seed is not None:
            self._python_rng = random.Random(seed)
            self.dag_generator.rng = self._python_rng
            np.random.seed(seed)

        self.current_time = 0.0
        self.reward_calculator.reset()

        self.dag = self.dag_generator.generate()
        self.dag.nodes[0]["available"] = 0
        self.dag.nodes[0]["scheduled_location"] = 0

        self._build_devices()

        self.scheduler = Scheduler(
            dag=self.dag,
            transmission_model=self.transmission_model,
            devices=self.devices,
            replay=self.mobility_replay if self.use_time_dependent_mobility else None,
        )

        self.scheduler.update_available_nodes()
        self.previous_makespan = 0.0
        self._init_buffers()

        return self._get_observation(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        terminated = False
        truncated = False
        invalid_reward = -10.0

        node_signal = float(action[0])
        location_signal = float(action[1])

        node_id = self._map_continuous_to_node(node_signal)
        location_id = self._map_continuous_to_location(location_signal)

        valid_nodes = self._get_valid_nodes()
        if len(valid_nodes) == 0:
            return (
                self._get_observation(),
                -1.0,
                True,
                truncated,
                {"invalid_reason": "no_valid_actions"},
            )

        if node_id not in self.dag.nodes:
            return (
                self._get_observation(),
                invalid_reward,
                False,
                truncated,
                {"invalid_reason": "unknown_node"},
            )

        if location_id not in self.devices:
            return (
                self._get_observation(),
                invalid_reward,
                False,
                truncated,
                {"invalid_reason": "unknown_location"},
            )

        if not self._is_schedulable(node_id):
            return (
                self._get_observation(),
                invalid_reward,
                False,
                truncated,
                {"invalid_reason": "node_not_schedulable"},
            )

        # advance mobility to current step time first
        self.current_time += self.mobility_time_step
        self._sync_mobility_to_time(self.current_time)

        self.scheduler.schedule_node(
            node_id=node_id,
            device_id=location_id,
        )

        self.scheduler.update_available_nodes()
        self._sync_location_buffers()

        reward, metrics = self.reward_calculator.compute_scaled_reward(
            scheduler=self.scheduler,
            dag=self.dag,
            devices=self.devices,
        )

        if self.scheduler.is_all_scheduled():
            terminated = True
            reward += self.reward_calculator.final_reward(
                self.scheduler,
                self.dag,
                self.devices,
            )

        info = self._get_info()
        info.update(metrics)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        print("\n========== DAG STATE ==========")
        for node in sorted(self.dag.nodes, key=node_sort_key):
            attr = self.dag.nodes[node]
            print(
                f"Node {node} | "
                f"avail={attr['available']} | "
                f"loc={attr['scheduled_location']}"
            )

        print("\n========== SCHEDULE ==========")
        for node, info in self.scheduler.node_schedule_info.items():
            print(
                f"Node {node} -> "
                f"Device {info['device_id']} | "
                f"EST={info['EST']:.4f} | "
                f"CT={info['CT']:.4f}"
            )