"""PyBullet surgical workspace with RCM-constrained planning and path optimisation."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pybullet as p

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import (
    path_cost,
    plan_rrt,
    shortcut_path,
)
from src.robotics.safety import evaluate_instrument_safety
from src.robotics.trajectory import interpolate_joint_trajectory


@dataclass(frozen=True)
class SceneObjectIds:
    """PyBullet object identifiers for the surgical scene."""

    rcm_id: int
    instrument_id: int
    target_id: int
    structure_ids: tuple[int, ...]
    safety_region_ids: tuple[int, ...]


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Quantitative metrics for an executed trajectory."""

    maximum_rcm_error: float
    minimum_surface_clearance: float
    minimum_safety_clearance: float
    collision: bool
    safety_margin_violation: bool


class SurgicalScene:
    """Manage the simulated minimally invasive surgical workspace."""

    def __init__(self, gui: bool = True) -> None:
        connection_mode = p.GUI if gui else p.DIRECT

        self.client_id = p.connect(connection_mode)

        if self.client_id < 0:
            raise RuntimeError(
                "Unable to connect to PyBullet."
            )

        p.resetSimulation(
            physicsClientId=self.client_id,
        )

        p.setGravity(
            0.0,
            0.0,
            0.0,
            physicsClientId=self.client_id,
        )

        self.rcm_position = np.zeros(
            3,
            dtype=float,
        )

        self.instrument_model = SurgicalInstrument(
            rcm_position=self.rcm_position,
        )

        self.instrument_radius = 0.006
        self.instrument_proximal_length = 0.10

        self.target_position = np.array(
            [0.24, 0.08, 0.03],
            dtype=float,
        )

        self.structures = (
            SphericalStructure(
                centre=np.array(
                    [0.14, 0.04, 0.00],
                    dtype=float,
                ),
                physical_radius=0.025,
                safety_margin=0.015,
            ),
            SphericalStructure(
                centre=np.array(
                    [0.18, -0.06, 0.02],
                    dtype=float,
                ),
                physical_radius=0.025,
                safety_margin=0.015,
            ),
        )

        self.object_ids: SceneObjectIds | None = None

        if gui:
            self._configure_camera()

    def _configure_camera(self) -> None:
        """Position the PyBullet viewing camera."""

        p.resetDebugVisualizerCamera(
            cameraDistance=0.55,
            cameraYaw=45.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[
                0.10,
                0.0,
                0.02,
            ],
            physicsClientId=self.client_id,
        )

    def close(self) -> None:
        """Disconnect from PyBullet."""

        if p.isConnected(self.client_id):
            p.disconnect(
                physicsClientId=self.client_id,
            )

    def _create_sphere(
        self,
        position: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> int:
        """Create a visual sphere."""

        position = np.asarray(
            position,
            dtype=float,
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=float(radius),
            rgbaColor=rgba,
            physicsClientId=self.client_id,
        )

        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.tolist(),
            physicsClientId=self.client_id,
        )

    def _cylinder_pose_between_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        float,
    ]:
        """Calculate cylinder pose between two 3-D points."""

        start = np.asarray(
            start,
            dtype=float,
        )

        end = np.asarray(
            end,
            dtype=float,
        )

        direction = end - start
        length = np.linalg.norm(direction)

        if np.isclose(length, 0.0):
            raise ValueError(
                "Cylinder start and end points cannot coincide."
            )

        midpoint = (
            start + end
        ) / 2.0

        unit_direction = (
            direction / length
        )

        z_axis = np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )

        rotation_axis = np.cross(
            z_axis,
            unit_direction,
        )

        axis_norm = np.linalg.norm(
            rotation_axis
        )

        dot_product = np.clip(
            np.dot(
                z_axis,
                unit_direction,
            ),
            -1.0,
            1.0,
        )

        angle = np.arccos(
            dot_product
        )

        if axis_norm < 1e-12:
            if dot_product >= 0.0:
                quaternion = (
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
            else:
                quaternion = tuple(
                    p.getQuaternionFromAxisAngle(
                        [1.0, 0.0, 0.0],
                        np.pi,
                    )
                )
        else:
            rotation_axis = (
                rotation_axis / axis_norm
            )

            quaternion = tuple(
                p.getQuaternionFromAxisAngle(
                    rotation_axis.tolist(),
                    float(angle),
                )
            )

        return (
            midpoint,
            quaternion,
            float(length),
        )

    def _create_cylinder_between_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> int:
        """Create a visual cylinder."""

        (
            midpoint,
            quaternion,
            length,
        ) = self._cylinder_pose_between_points(
            start,
            end,
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=float(radius),
            length=length,
            rgbaColor=rgba,
            physicsClientId=self.client_id,
        )

        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=midpoint.tolist(),
            baseOrientation=quaternion,
            physicsClientId=self.client_id,
        )

    def build_default_scene(
        self,
        initial_q: np.ndarray,
    ) -> SceneObjectIds:
        """Build the visible surgical workspace."""

        proximal, tip = (
            self.instrument_model.shaft_segment(
                initial_q,
                proximal_length=(
                    self.instrument_proximal_length
                ),
            )
        )

        rcm_id = self._create_sphere(
            position=self.rcm_position,
            radius=0.010,
            rgba=(
                1.0,
                0.55,
                0.0,
                1.0,
            ),
        )

        instrument_id = (
            self._create_cylinder_between_points(
                start=proximal,
                end=tip,
                radius=self.instrument_radius,
                rgba=(
                    0.65,
                    0.65,
                    0.65,
                    1.0,
                ),
            )
        )

        target_id = self._create_sphere(
            position=self.target_position,
            radius=0.012,
            rgba=(
                0.0,
                0.9,
                0.2,
                1.0,
            ),
        )

        structure_ids: list[int] = []
        safety_region_ids: list[int] = []

        for structure in self.structures:
            safety_region_ids.append(
                self._create_sphere(
                    position=structure.centre,
                    radius=structure.safety_radius,
                    rgba=(
                        1.0,
                        0.25,
                        0.25,
                        0.18,
                    ),
                )
            )

            structure_ids.append(
                self._create_sphere(
                    position=structure.centre,
                    radius=structure.physical_radius,
                    rgba=(
                        0.9,
                        0.05,
                        0.05,
                        1.0,
                    ),
                )
            )

        self.object_ids = SceneObjectIds(
            rcm_id=rcm_id,
            instrument_id=instrument_id,
            target_id=target_id,
            structure_ids=tuple(
                structure_ids
            ),
            safety_region_ids=tuple(
                safety_region_ids
            ),
        )

        return self.object_ids

    def update_instrument(
        self,
        q: np.ndarray,
    ) -> None:
        """Update the visible instrument."""

        if self.object_ids is None:
            raise RuntimeError(
                "Scene must be built first."
            )

        proximal, tip = (
            self.instrument_model.shaft_segment(
                q,
                proximal_length=(
                    self.instrument_proximal_length
                ),
            )
        )

        (
            midpoint,
            quaternion,
            _,
        ) = self._cylinder_pose_between_points(
            proximal,
            tip,
        )

        p.resetBasePositionAndOrientation(
            self.object_ids.instrument_id,
            midpoint.tolist(),
            quaternion,
            physicsClientId=self.client_id,
        )


def densify_path(
    path: np.ndarray,
    samples_per_edge: int = 40,
) -> np.ndarray:
    """Convert sparse planner waypoints into a dense trajectory."""

    path = np.asarray(
        path,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 4
    ):
        raise ValueError(
            "path must have shape (N, 4)."
        )

    if len(path) < 2:
        raise ValueError(
            "path must contain at least two configurations."
        )

    if samples_per_edge < 2:
        raise ValueError(
            "samples_per_edge must be at least 2."
        )

    segments: list[np.ndarray] = []

    for index in range(
        len(path) - 1
    ):
        segment = interpolate_joint_trajectory(
            path[index],
            path[index + 1],
            num_steps=samples_per_edge,
        )

        if index > 0:
            segment = segment[1:]

        segments.append(
            segment
        )

    return np.vstack(
        segments
    )


def evaluate_trajectory(
    instrument: SurgicalInstrument,
    trajectory: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
) -> TrajectoryMetrics:
    """Evaluate RCM error, clearance and safety along a trajectory."""

    maximum_rcm_error = 0.0

    minimum_surface_clearance = float(
        "inf"
    )

    minimum_safety_clearance = float(
        "inf"
    )

    collision_detected = False

    safety_violation_detected = False

    for q in trajectory:
        proximal, tip = (
            instrument.shaft_segment(
                q,
                proximal_length=proximal_length,
            )
        )

        rcm_error = (
            instrument.rcm_error(q)
        )

        safety = evaluate_instrument_safety(
            shaft_start=proximal,
            shaft_end=tip,
            structures=structures,
            instrument_radius=instrument_radius,
        )

        maximum_rcm_error = max(
            maximum_rcm_error,
            rcm_error,
        )

        minimum_surface_clearance = min(
            minimum_surface_clearance,
            safety.minimum_surface_clearance,
        )

        minimum_safety_clearance = min(
            minimum_safety_clearance,
            safety.minimum_safety_clearance,
        )

        collision_detected = (
            collision_detected
            or safety.collision
        )

        safety_violation_detected = (
            safety_violation_detected
            or safety.safety_margin_violation
        )

    return TrajectoryMetrics(
        maximum_rcm_error=maximum_rcm_error,
        minimum_surface_clearance=(
            minimum_surface_clearance
        ),
        minimum_safety_clearance=(
            minimum_safety_clearance
        ),
        collision=collision_detected,
        safety_margin_violation=(
            safety_violation_detected
        ),
    )


def main() -> None:
    """Compare raw and smoothed RRT paths and visualise the optimised path."""

    start_q = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ],
        dtype=float,
    )

    goal_q = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ],
        dtype=float,
    )

    planning_scene = SurgicalScene(
        gui=False
    )

    planning_start = time.perf_counter()

    planning_result = plan_rrt(
        instrument=(
            planning_scene.instrument_model
        ),
        start_q=start_q,
        goal_q=goal_q,
        structures=(
            planning_scene.structures
        ),
        instrument_radius=(
            planning_scene.instrument_radius
        ),
        proximal_length=(
            planning_scene.instrument_proximal_length
        ),
        max_iterations=10000,
        step_size=0.08,
        goal_bias=0.20,
        edge_resolution=30,
        seed=7,
    )

    planning_time = (
        time.perf_counter()
        - planning_start
    )

    if not planning_result.success:
        planning_scene.close()

        raise RuntimeError(
            "RRT could not find a safe trajectory."
        )

    raw_path = planning_result.path

    smoothing_start = time.perf_counter()

    smoothed_path = shortcut_path(
        instrument=(
            planning_scene.instrument_model
        ),
        path=raw_path,
        structures=(
            planning_scene.structures
        ),
        instrument_radius=(
            planning_scene.instrument_radius
        ),
        proximal_length=(
            planning_scene.instrument_proximal_length
        ),
        edge_resolution=40,
        attempts=500,
        seed=11,
    )

    smoothing_time = (
        time.perf_counter()
        - smoothing_start
    )

    raw_trajectory = densify_path(
        raw_path,
        samples_per_edge=40,
    )

    smoothed_trajectory = densify_path(
        smoothed_path,
        samples_per_edge=40,
    )

    raw_metrics = evaluate_trajectory(
        instrument=(
            planning_scene.instrument_model
        ),
        trajectory=raw_trajectory,
        structures=(
            planning_scene.structures
        ),
        instrument_radius=(
            planning_scene.instrument_radius
        ),
        proximal_length=(
            planning_scene.instrument_proximal_length
        ),
    )

    smoothed_metrics = evaluate_trajectory(
        instrument=(
            planning_scene.instrument_model
        ),
        trajectory=smoothed_trajectory,
        structures=(
            planning_scene.structures
        ),
        instrument_radius=(
            planning_scene.instrument_radius
        ),
        proximal_length=(
            planning_scene.instrument_proximal_length
        ),
    )

    raw_cost = path_cost(
        raw_path
    )

    smoothed_cost = path_cost(
        smoothed_path
    )

    waypoint_reduction = (
        100.0
        * (
            len(raw_path)
            - len(smoothed_path)
        )
        / len(raw_path)
    )

    cost_reduction = (
        100.0
        * (
            raw_cost
            - smoothed_cost
        )
        / raw_cost
    )

    print()
    print("RRT + path optimisation")
    print("-----------------------")
    print(
        f"Planning success: "
        f"{planning_result.success}"
    )
    print(
        f"RRT iterations: "
        f"{planning_result.iterations}"
    )
    print(
        f"Planning time: "
        f"{planning_time:.6f} s"
    )
    print(
        f"Smoothing time: "
        f"{smoothing_time:.6f} s"
    )

    print()
    print("Raw RRT path")
    print("------------")
    print(
        f"Waypoints: {len(raw_path)}"
    )
    print(
        f"Path cost: {raw_cost:.6f}"
    )
    print(
        "Minimum physical clearance: "
        f"{raw_metrics.minimum_surface_clearance:.6f} m"
    )
    print(
        "Minimum safety clearance: "
        f"{raw_metrics.minimum_safety_clearance:.6f} m"
    )
    print(
        "Maximum RCM error: "
        f"{raw_metrics.maximum_rcm_error:.12e} m"
    )
    print(
        f"Collision: "
        f"{raw_metrics.collision}"
    )
    print(
        "Safety-margin violation: "
        f"{raw_metrics.safety_margin_violation}"
    )

    print()
    print("Smoothed RRT path")
    print("-----------------")
    print(
        f"Waypoints: {len(smoothed_path)}"
    )
    print(
        f"Path cost: {smoothed_cost:.6f}"
    )
    print(
        "Minimum physical clearance: "
        f"{smoothed_metrics.minimum_surface_clearance:.6f} m"
    )
    print(
        "Minimum safety clearance: "
        f"{smoothed_metrics.minimum_safety_clearance:.6f} m"
    )
    print(
        "Maximum RCM error: "
        f"{smoothed_metrics.maximum_rcm_error:.12e} m"
    )
    print(
        f"Collision: "
        f"{smoothed_metrics.collision}"
    )
    print(
        "Safety-margin violation: "
        f"{smoothed_metrics.safety_margin_violation}"
    )

    print()
    print("Optimisation improvement")
    print("------------------------")
    print(
        "Waypoint reduction: "
        f"{waypoint_reduction:.2f}%"
    )
    print(
        "Path-cost reduction: "
        f"{cost_reduction:.2f}%"
    )

    planning_scene.close()

    scene = SurgicalScene(
        gui=True
    )

    scene.build_default_scene(
        initial_q=(
            smoothed_trajectory[0]
        ),
    )

    try:
        for q in smoothed_trajectory:
            if not p.isConnected(
                scene.client_id
            ):
                break

            scene.update_instrument(q)

            p.stepSimulation(
                physicsClientId=(
                    scene.client_id
                ),
            )

            time.sleep(
                1.0 / 60.0
            )

        while p.isConnected(
            scene.client_id
        ):
            p.stepSimulation(
                physicsClientId=(
                    scene.client_id
                ),
            )

            time.sleep(
                1.0 / 60.0
            )

    except KeyboardInterrupt:
        pass

    finally:
        scene.close()


if __name__ == "__main__":
    main()