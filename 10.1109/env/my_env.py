try:
    from env.dag_generator import DAGGenerator
    from env.reward import RewardCalculator
    from env.scheduler import ComputeNode, Scheduler
    from env.transmission import TransmissionModel
except ImportError:
    from dag_generator import DAGGenerator
    from reward import RewardCalculator
    from scheduler import ComputeNode, Scheduler
    from transmission import TransmissionModel

import random
import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------
# help functions
# --------------

def node_sort_key(node):
    
    if node == 0:
        return (0, -1, -1)

    if isinstance(node, tuple):
        task_id, node_id = node
        return (1, task_id, node_id)

    return (2, node, node)

class VehicleMobility:
    """Simple continuous mobility model for a vehicle."""
    def __init__(self, x, y, world_size, rng):
        self.rng = rng
        self.x = float(x)
        self.y = float(y)
        self.world_size = world_size

        self.velocity = self.rng.uniform(5.0, 15.0)
        self.acceleration = self.rng.uniform(-0.5, 0.5)
        self.heading = self.rng.uniform(0.0, 2 * np.pi)
        self.angular_velocity = self.rng.uniform(-0.05, 0.05)
    def step(self, dt=1.0):
        self.acceleration += self.rng.normal(0.0, 0.05)
        self.acceleration = np.clip(self.acceleration, -1.0, 1.0)

        self.angular_velocity += self.rng.normal(0.0, 0.005)
        self.angular_velocity = np.clip(self.angular_velocity, -0.1, 0.1)

        self.heading += self.angular_velocity * dt
        self.velocity += self.acceleration * dt
        self.velocity = np.clip(self.velocity, 1.0, 25.0)

        self.x += np.cos(self.heading) * self.velocity * dt
        self.y += np.sin(self.heading) * self.velocity * dt

        if self.x < 0 or self.x > self.world_size:
            self.heading = np.pi - self.heading
            self.x = np.clip(self.x, 0, self.world_size)

        if self.y < 0 or self.y > self.world_size:
            self.heading = -self.heading
            self.y = np.clip(self.y, 0, self.world_size)

class VEC(gym.Env):
    """
    main body of paper
    """
    def __init__(self, num_ve:int = 5, num_ves:int = 1, num_nodes:int = 5):
        
        super(VEC, self).__init__()
        self.num_ve = num_ve
        self.num_ves = num_ves
        self.num_nodes = num_nodes
        self.world_size = 1000
        self.num_locations = None
        self._num_task_graph_nodes = num_ve * (num_nodes + 1) + 1
        self.dag_generator = DAGGenerator(
            num_tasks=num_ve,
            num_nodes=num_nodes,
            max_out_degree=min(5,num_nodes),
        )

        self.reward_calculator = RewardCalculator()
        self.previous_makespan = 0.0


        self.transmission_model = TransmissionModel()
        self.devices = {}
        self.dag = self.dag_generator.generate()
        self._build_devices()
        self.mobility_models = {}
        assert self.num_locations is not None
        self.scheduler = Scheduler(
            dag=self.dag,
            transmission_model=self.transmission_model,
            devices=self.devices,
        )


        self.node_feature_dim = 6
        """
        because:
        self.dag.nodes[node]["cpu_cycles"],
        self.dag.nodes[node]["data_size"],
        self.dag.nodes[node]["in_degree"],
        self.dag.nodes[node]["out_degree"],
        self.dag.nodes[node]["scheduled_location"],
        self.dag.nodes[node]["available"],
        """
        self.total_actions = self._num_task_graph_nodes * self.num_locations
        self.action_space = spaces.Box(low=-1.0, high=1.0,
            shape=(self.total_actions,),
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
                    shape=(self.num_ve, 5),
                    dtype=np.float32,
                ),
                "locations": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_locations, 6),
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
        
        self.reset()

    def _build_devices(self):
        self.devices = {}
        self.mobility_models = {}

        for vehicle_id in range(self.num_ve):
            x = np.random.uniform(0, self.world_size)
            y = np.random.uniform(0, self.world_size)

            self.devices[vehicle_id] = ComputeNode(
                node_id=vehicle_id,
                compute_power=1e9,
                num_processors=1,
                x=x,
                y=y,
                node_type="VE",
            )
            self.mobility_models[vehicle_id] = VehicleMobility(
                x=float(x),
                y=float(y),
                world_size=self.world_size,
                rng=np.random,
            )

        for ves_id in range(self.num_ves):
            device_id = self.num_ve + ves_id
            self.devices[device_id] = ComputeNode(
                node_id=device_id,
                compute_power=10e9,
                num_processors=4,
                x=float(np.random.uniform(0, self.world_size)),
                y=float(np.random.uniform(0, self.world_size)),
                node_type="VES",
            )

        self.num_locations = len(self.devices)

    def reset(self, seed=None): # type: ignore # مطمئن نیستم چرا گیر می‌داد به این
        super().reset(seed=seed)

        self._build_devices()
        self.dag = self.dag_generator.generate()
        self.dag.nodes[0]["available"] = 0
        self.dag.nodes[0]["scheduled_location"] = 0
        self.scheduler = Scheduler(
            dag=self.dag,
            transmission_model=self.transmission_model,
            devices=self.devices,
        )
        self.scheduler.update_available_nodes()
        self.previous_makespan = 0.0

        return self._get_observation(), self._get_info()

    def _decode_action_node(self, flat_index)->tuple | int:
        if flat_index == 0:
            return 0

        task_id = (flat_index - 1) // (self.num_nodes + 1)
        node_id = (flat_index - 1) % (self.num_nodes + 1) + 1
        return (task_id, node_id)

    def _encode_node(self, node) -> int:
        if node == 0:
            return 0

        task_id, node_id = node
        return 1 + task_id * (self.num_nodes + 1) + (node_id - 1)

    def action_masks(self):
        mask = np.zeros(self.total_actions, dtype=bool)

        for node in self.dag.nodes:
            if not self._is_schedulable(node):
                continue

            node_idx = self._encode_node(node)

            start = node_idx * self.num_locations # type: ignore
            end = start + self.num_locations # type: ignore

            mask[start:end] = True

        return mask

    def _is_schedulable(self, node):
        if node == 0:
            return False

        attr = self.dag.nodes[node]
        return attr["available"] == 1 and attr["scheduled_location"] == -1

# still not sure
    def _get_info(self):
        return {
            "action_mask": self.action_masks(),
            "node_action_mask": np.flatnonzero(self.action_masks()),
            "makespan": self.reward_calculator.calculate_makespan(self.scheduler),
        }

    def _update_vehicle_positions(self):
        for device_id, mobility in self.mobility_models.items():
            mobility.step(dt=1.0)
            self.devices[device_id].x = mobility.x
            self.devices[device_id].y = mobility.y

    
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
            [
                [
                    self.mobility_models[vehicle_id].x,
                    self.mobility_models[vehicle_id].y,
                    self.mobility_models[vehicle_id].velocity,
                    self.mobility_models[vehicle_id].acceleration,
                    self.mobility_models[vehicle_id].heading,
                ]
                for vehicle_id in range(self.num_ve)
            ],
            dtype=np.float32,
        )

        locations = []
        node_runtime = []
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

            locations.append(
                [
                    1.0,
                    task_count,
                    device.x,
                    device.y,
                    device.compute_power,
                    current_finish_time,
                ]
            )
            node_runtime.append([device.compute_power, current_finish_time])

        return {
            "node_features": node_features,
            "adj_matrix": adj,
            "trajectory": trajectory,
            "locations": np.array(locations, dtype=np.float32),
            "node_runtime": np.array(node_runtime, dtype=np.float32),
        } 
    def step(self, action):

        action = np.asarray(action, dtype=np.float32)

        terminated = False
        truncated = False
        invalid_reward = -1.0

        mask = self.action_masks()

        if not np.any(mask):
            return (
                self._get_observation(),
                -1.0,
                True,
                truncated,
                {"invalid_reason": "no_valid_actions"},
            )

        # مهم‌ترین خط: masking واقعی continuous space
        masked_action = np.where(mask, action, -np.inf)

        flat_action = int(np.argmax(masked_action))

        flat_node_id = flat_action // self.num_locations # type: ignore
        location_id = flat_action % self.num_locations # type: ignore

        node_id = self._decode_action_node(flat_node_id)

        if node_id not in self.dag.nodes:
            return self._get_observation(), invalid_reward, False, truncated, {"invalid_reason": "unknown_node"}

        if location_id not in self.devices:
            return self._get_observation(), invalid_reward, False, truncated, {"invalid_reason": "unknown_location"}

        if not self._is_schedulable(node_id):
            return self._get_observation(), invalid_reward, False, truncated, {"invalid_reason": "node_not_schedulable"}

        self._update_vehicle_positions()

        self.scheduler.schedule_node(node_id=node_id, device_id=location_id)
        self.scheduler.update_available_nodes()

        current_makespan = self.reward_calculator.calculate_makespan(self.scheduler)

        local_baseline = self.reward_calculator.estimate_local_completion(
            self.dag,
            self.devices,
        )

        reward = self.reward_calculator.compute_scaled_reward(
            previous_makespan=self.previous_makespan,
            current_makespan=current_makespan,
            local_baseline=local_baseline,
        )

        self.previous_makespan = current_makespan

        if self.scheduler.is_all_scheduled():
            terminated = True
            reward += self.reward_calculator.final_reward(
                self.scheduler,
                self.dag,
                self.devices,
            )

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    def get_valid_actions(self):
        return np.flatnonzero(self.action_masks())
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