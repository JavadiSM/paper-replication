from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class MobilityState:
    vehicle_id: Union[int, str]
    trace_id: str
    time_ms: float
    x: float
    y: float
    speed: float
    heading_rad: float
    heading_deg: float
    acceleration: float


@dataclass(frozen=True)
class VehicleAssignment:
    device_id: Union[int, str]
    trace_id: str
    valid_from_ms: float
    valid_to_ms: float

    def active_at(self, time_ms: float) -> bool:
        return self.valid_from_ms <= time_ms <= self.valid_to_ms


class MobilityReplay:
    """
    Preprocess dataset once, then:
      1) register/bind devices
      2) resolve device -> trace_id at time t
      3) query trace state by trace_id and time t

    VE:
      - mapping is needed
      - trace may expire, so remap can happen at runtime

    VES:
      - no mapping needed
      - use its own fixed x, y
    """

    def __init__(
        self,
        dataset_path: str,
        time_scale_ms: float = 1.0,
        skip_header: bool = True,
        clamp_out_of_range: bool = True,
    ):
        self.dataset_path = dataset_path
        self.time_scale_ms = float(time_scale_ms)
        self.skip_header = skip_header
        self.clamp_out_of_range = clamp_out_of_range

        self.origin_ms = 0.0
        self.global_time_min = 0.0

        # trace_id -> arrays
        self.traces: Dict[str, Dict[str, np.ndarray]] = {}

        # trace_id -> (start_ms, end_ms) in scaled time
        self.trace_bounds: Dict[str, tuple[float, float]] = {}

        # device_id -> assignment
        self.mapping: Dict[Union[int, str], VehicleAssignment] = {}

        # device_id -> node_type
        self.device_types: Dict[Union[int, str], str] = {}

        self.preprocess()

    def preprocess(self) -> None:
        self.traces, self.trace_bounds = self._load_dataset()

    def _load_dataset(self) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, tuple[float, float]]]:
        raw = defaultdict(lambda: {"t": [], "x": [], "y": [], "speed": [], "angle": []})
        global_min_t = None

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            if self.skip_header:
                next(f, None)

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                try:
                    t = float(parts[0])
                    vid = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    speed = float(parts[4])
                    angle = float(parts[5])
                except ValueError:
                    continue

                if global_min_t is None or t < global_min_t:
                    global_min_t = t

                raw[vid]["t"].append(t)
                raw[vid]["x"].append(x)
                raw[vid]["y"].append(y)
                raw[vid]["speed"].append(speed)
                raw[vid]["angle"].append(angle)

        if global_min_t is None:
            self.global_time_min = 0.0
            return {}, {}

        self.global_time_min = float(global_min_t)

        traces: Dict[str, Dict[str, np.ndarray]] = {}
        bounds: Dict[str, tuple[float, float]] = {}

        for vid, data in raw.items():
            t = np.asarray(data["t"], dtype=np.float64)
            x = np.asarray(data["x"], dtype=np.float64)
            y = np.asarray(data["y"], dtype=np.float64)
            speed = np.asarray(data["speed"], dtype=np.float64)
            angle = np.asarray(data["angle"], dtype=np.float64)

            if len(t) == 0:
                continue

            order = np.argsort(t)
            t = t[order]
            x = x[order]
            y = y[order]
            speed = speed[order]
            angle = angle[order]

            t_scaled = (t - self.global_time_min) * self.time_scale_ms

            traces[vid] = {
                "t": t_scaled,
                "x": x,
                "y": y,
                "speed": speed,
                "angle_deg": angle,
            }
            bounds[vid] = (float(t_scaled[0]), float(t_scaled[-1]))

        return traces, bounds

    def set_zero(self, origin_ms: float) -> None:
        self.origin_ms = float(origin_ms)

    def reset_zero(self) -> None:
        self.origin_ms = 0.0

    def replay_time(self, query_time_ms: float) -> float:
        return float(query_time_ms) + self.origin_ms

    def _device_id(self, device_or_id) -> Union[int, str]:
        return device_or_id.node_id if hasattr(device_or_id, "node_id") else device_or_id

    def _device_type(self, device_or_id) -> Optional[str]:
        return getattr(device_or_id, "node_type", None)

    def _active_traces_at(self, query_time_ms: float) -> List[str]:
        return [
            trace_id
            for trace_id, (t0, t1) in self.trace_bounds.items()
            if t0 <= query_time_ms <= t1
        ]

    def bind_devices(self, devices: Dict[Union[int, str], object], rng, time_ms: float = 0.0) -> Dict[Union[int, str], VehicleAssignment]:
        """
        Register devices once.
        VE devices get a trace assignment.
        VES devices are only registered and do not need mapping.
        """
        self.device_types = {}
        self.mapping = {}

        query_time = self.replay_time(time_ms)
        active_traces = self._active_traces_at(query_time)

        if not active_traces:
            raise RuntimeError(f"No active traces at time {query_time}.")

        used: set[str] = set()

        for device in devices.values():
            device_id = self._device_id(device)
            device_type = self._device_type(device) or "VE"
            self.device_types[device_id] = device_type

            if device_type == "VES":
                continue

            candidates = [tid for tid in active_traces if tid not in used]
            if not candidates:
                candidates = active_traces[:]

            if not candidates:
                continue

            trace_id = rng.choice(candidates)
            t0, t1 = self.trace_bounds[trace_id]

            self.mapping[device_id] = VehicleAssignment(
                device_id=device_id,
                trace_id=trace_id,
                valid_from_ms=t0,
                valid_to_ms=t1,
            )
            used.add(trace_id)

        return self.mapping

    def refresh_mapping(self, time_ms: float, rng) -> Dict[Union[int, str], VehicleAssignment]:
        """
        Refresh only VE mappings. If a trace is no longer valid, remap it.
        If no active trace exists for a VE, that VE is removed from mapping.
        """
        query_time = self.replay_time(time_ms)
        active_traces = self._active_traces_at(query_time)

        if not active_traces:
            self.mapping = {}
            return self.mapping

        new_mapping: Dict[Union[int, str], VehicleAssignment] = {}
        used: set[str] = set()

        for device_id, device_type in self.device_types.items():
            if device_type == "VES":
                continue

            old = self.mapping.get(device_id)
            if old is not None and old.active_at(query_time):
                new_mapping[device_id] = old
                used.add(old.trace_id)
                continue

            candidates = [tid for tid in active_traces if tid not in used]
            if not candidates:
                candidates = active_traces[:]

            if not candidates:
                continue

            trace_id = rng.choice(candidates)
            t0, t1 = self.trace_bounds[trace_id]
            new_mapping[device_id] = VehicleAssignment(
                device_id=device_id,
                trace_id=trace_id,
                valid_from_ms=t0,
                valid_to_ms=t1,
            )
            used.add(trace_id)

        self.mapping = new_mapping
        return self.mapping

    def resolve_trace_id(self, device_or_id, time_ms: float, rng=None) -> Optional[str]:
        """
        Return the dataset trace_id for a VE at time_ms.
        For VES, returns None.
        If the current mapping is invalid and rng is provided, a remap is attempted.
        """
        device_id = self._device_id(device_or_id)
        device_type = self.device_types.get(device_id, self._device_type(device_or_id) or "VE")

        if device_type == "VES":
            return None

        query_time = self.replay_time(time_ms)
        assignment = self.mapping.get(device_id)

        if assignment is not None and assignment.active_at(query_time):
            return assignment.trace_id

        if rng is None:
            return None

        active_traces = self._active_traces_at(query_time)
        if not active_traces:
            return None

        used = {ass.trace_id for did, ass in self.mapping.items() if did != device_id and ass.active_at(query_time)}
        candidates = [tid for tid in active_traces if tid not in used]
        if not candidates:
            candidates = active_traces[:]

        if not candidates:
            return None

        trace_id = rng.choice(candidates)
        t0, t1 = self.trace_bounds[trace_id]
        self.mapping[device_id] = VehicleAssignment(
            device_id=device_id,
            trace_id=trace_id,
            valid_from_ms=t0,
            valid_to_ms=t1,
        )
        return trace_id

    def get_state(self, trace_id: str, time_ms: float) -> MobilityState:
        """
        Direct trace lookup by dataset id.
        """
        if trace_id not in self.traces:
            raise KeyError(f"Unknown trace_id: {trace_id}")

        tr = self.traces[trace_id]
        t = tr["t"]
        x = tr["x"]
        y = tr["y"]
        speed = tr["speed"]
        angle_deg = tr["angle_deg"]

        if len(t) == 0:
            raise RuntimeError(f"Empty trace for '{trace_id}'.")

        query_time = self.replay_time(time_ms)

        if len(t) == 1:
            hdg_deg = float(angle_deg[0])
            return MobilityState(
                vehicle_id=trace_id,
                trace_id=trace_id,
                time_ms=query_time,
                x=float(x[0]),
                y=float(y[0]),
                speed=float(speed[0]),
                heading_rad=float(np.deg2rad(hdg_deg)),
                heading_deg=hdg_deg,
                acceleration=0.0,
            )

        if query_time <= t[0]:
            if not self.clamp_out_of_range:
                raise ValueError(f"time_ms={query_time} is before trace start {t[0]}")
            i0, i1 = 0, 1
            alpha = 0.0
        elif query_time >= t[-1]:
            if not self.clamp_out_of_range:
                raise ValueError(f"time_ms={query_time} is after trace end {t[-1]}")
            i0, i1 = len(t) - 2, len(t) - 1
            alpha = 1.0
        else:
            i1 = int(np.searchsorted(t, query_time, side="right"))
            i0 = i1 - 1
            t0 = float(t[i0])
            t1 = float(t[i1])
            alpha = 0.0 if t1 == t0 else (query_time - t0) / (t1 - t0)

        xq = float(x[i0] + alpha * (x[i1] - x[i0]))
        yq = float(y[i0] + alpha * (y[i1] - y[i0]))
        vq = float(speed[i0] + alpha * (speed[i1] - speed[i0]))

        r0 = np.deg2rad(float(angle_deg[i0]))
        r1 = np.deg2rad(float(angle_deg[i1]))
        c0, s0 = np.cos(r0), np.sin(r0)
        c1, s1 = np.cos(r1), np.sin(r1)
        c = c0 + alpha * (c1 - c0)
        s = s0 + alpha * (s1 - s0)
        hdg_deg = float(np.rad2deg(np.arctan2(s, c)))
        hdg_rad = float(np.deg2rad(hdg_deg))

        dt = float(t[i1] - t[i0])
        acc = 0.0 if dt <= 0.0 else float((speed[i1] - speed[i0]) / dt)

        return MobilityState(
            vehicle_id=trace_id,
            trace_id=trace_id,
            time_ms=query_time,
            x=xq,
            y=yq,
            speed=vq,
            heading_rad=hdg_rad,
            heading_deg=hdg_deg,
            acceleration=acc,
        )