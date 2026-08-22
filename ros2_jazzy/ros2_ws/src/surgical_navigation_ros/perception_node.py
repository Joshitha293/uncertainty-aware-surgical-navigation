"""ROS 2 node integrating task-aware surgical active perception."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.task_relevance import (
    SurgicalTask,
    TaskRelevanceConfig,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScoringConfig,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
)


def rotation_from_forward(
    forward: Sequence[float],
) -> np.ndarray:
    """Construct a rotation matrix whose forward axis follows the target."""

    forward_vector = np.asarray(
        forward,
        dtype=float,
    )

    norm = np.linalg.norm(
        forward_vector
    )

    if norm <= 0.0:
        raise ValueError(
            "Forward vector must be non-zero."
        )

    forward_vector = (
        forward_vector / norm
    )

    reference = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    if abs(
        np.dot(
            forward_vector,
            reference,
        )
    ) > 0.95:
        reference = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )

    right = np.cross(
        reference,
        forward_vector,
    )

    right_norm = np.linalg.norm(
        right
    )

    right = (
        right / right_norm
    )

    up = np.cross(
        forward_vector,
        right,
    )

    up_norm = np.linalg.norm(
        up
    )

    up = (
        up / up_norm
    )

    return np.column_stack(
        (
            right,
            up,
            forward_vector,
        )
    )


class TaskAwarePerceptionNode(Node):
    """ROS 2 integration layer for task-aware active perception."""

    def __init__(self) -> None:
        super().__init__(
            "task_aware_perception_node"
        )

        self.publisher = (
            self.create_publisher(
                String,
                "perception/task_aware_selection",
                10,
            )
        )

        self.timer = (
            self.create_timer(
                2.0,
                self.run_perception_cycle,
            )
        )

        self.controller = (
            self._build_controller()
        )

        self.target = (
            SphericalStructure(
                centre=np.array(
                    [0.14, 0.04, 0.00],
                    dtype=float,
                ),
                physical_radius=0.025,
                safety_margin=0.015,
            )
        )

        self.occluders = (
            (
                SphericalStructure(
                    centre=np.array(
                        [0.18, -0.06, 0.02],
                        dtype=float,
                    ),
                    physical_radius=0.025,
                    safety_margin=0.015,
                ),
            )
        )

        self.current_pose = (
            self._make_pose(
                position=np.array(
                    [0.10, 0.10, 0.18],
                    dtype=float,
                ),
                target=self.target.centre,
            )
        )

        self.candidates = (
            self._build_candidates()
        )

        self.get_logger().info(
            "Task-aware surgical perception "
            "ROS 2 node started."
        )

    def _build_controller(
        self,
    ) -> TaskAwareActivePerception:
        """Construct the validated Day 7 task-aware controller."""

        intrinsics = CameraIntrinsics(
            horizontal_fov=np.deg2rad(
                70.0
            ),
            vertical_fov=np.deg2rad(
                50.0
            ),
            near_distance=0.01,
            far_distance=1.0,
        )

        camera = SurgicalCamera(
            intrinsics=intrinsics
        )

        observation_config = (
            ObservationModelConfig(
                base_sigma=0.002,
                reference_distance=0.15,
                distance_weight=1.0,
                angle_weight=1.0,
                invisible_sigma=0.05,
                occluded_sigma=0.03,
            )
        )

        observation_model = (
            ViewpointObservationModel(
                camera=camera,
                config=observation_config,
            )
        )

        scoring_config = (
            ViewpointScoringConfig(
                uncertainty_weight=1.0,
                movement_weight=0.1,
                occlusion_penalty=2.0,
                invisibility_penalty=4.0,
            )
        )

        generic_scorer = (
            GenericViewpointScorer(
                observation_model=(
                    observation_model
                ),
                config=scoring_config,
            )
        )

        trajectory = np.array(
            [
                [0.05, 0.00, 0.05],
                [0.10, 0.02, 0.08],
                [0.14, 0.04, 0.00],
            ],
            dtype=float,
        )

        safety_critical_points = (
            np.array(
                [
                    [0.14, 0.04, 0.00],
                    [0.15, 0.04, 0.01],
                ],
                dtype=float,
            )
        )

        task = SurgicalTask(
            trajectory=trajectory,
            safety_critical_points=(
                safety_critical_points
            ),
        )

        task_config = (
            TaskAwareScoringConfig(
                task_weight=2.0,
                alignment_weight=1.0,
                uncertainty_weight=1.0,
            )
        )

        relevance_config = (
            TaskRelevanceConfig(
                relevance_sigma=0.03,
                minimum_relevance=0.0,
                maximum_relevance=1.0,
            )
        )

        task_aware_scorer = (
            TaskAwareViewpointScorer(
                generic_scorer=generic_scorer,
                task=task,
                task_config=task_config,
                relevance_config=relevance_config,
            )
        )

        return TaskAwareActivePerception(
            scorer=task_aware_scorer
        )

    def _make_pose(
        self,
        position: np.ndarray,
        target: np.ndarray,
    ) -> CameraPose:
        """Create a camera pose looking toward a target."""

        direction = (
            target
            - position
        )

        rotation = (
            rotation_from_forward(
                direction
            )
        )

        return CameraPose(
            position=np.asarray(
                position,
                dtype=float,
            ),
            rotation=rotation,
        )

    def _build_candidates(
        self,
    ) -> tuple[
        CandidateViewpoint,
        ...,
    ]:
        """Create a deterministic set of candidate viewpoints."""

        target = np.asarray(
            self.target.centre,
            dtype=float,
        )

        positions = (
            np.array(
                [
                    [0.08, 0.04, 0.16],
                    [0.12, 0.10, 0.12],
                    [0.22, 0.10, 0.12],
                    [0.26, 0.02, 0.10],
                    [0.16, -0.12, 0.12],
                    [0.08, -0.08, 0.18],
                ],
                dtype=float,
            )
        )

        candidates = []

        for index, position in enumerate(
            positions
        ):
            pose = self._make_pose(
                position=position,
                target=target,
            )

            candidates.append(
                CandidateViewpoint(
                    pose=pose,
                    radius=float(
                        np.linalg.norm(
                            position
                            - target
                        )
                    ),
                    azimuth=float(
                        np.arctan2(
                            position[1]
                            - target[1],
                            position[0]
                            - target[0],
                        )
                    ),
                    elevation=float(
                        np.arctan2(
                            position[2]
                            - target[2],
                            np.linalg.norm(
                                position[:2]
                                - target[:2]
                            ),
                        )
                    ),
                )
            )

        return tuple(
            candidates
        )

    def run_perception_cycle(
        self,
    ) -> None:
        """Run one task-aware viewpoint-selection cycle."""

        try:
            result = (
                self.controller.select_viewpoint(
                    current_pose=self.current_pose,
                    candidates=self.candidates,
                    target=self.target,
                    occluders=self.occluders,
                )
            )

            selected_position = (
                result.selected_position
            )

            payload = {
                "selected_position": [
                    float(value)
                    for value in selected_position
                ],
                "candidate_count": int(
                    result.candidate_count
                ),
                "task_relevance": float(
                    result.task_relevance
                ),
                "task_alignment": float(
                    result.task_alignment
                ),
                "task_uncertainty_m": float(
                    result.task_uncertainty
                ),
                "task_aware_score": float(
                    result.selected_score.score
                ),
                "generic_score": float(
                    result.selected_score
                    .generic_score.score
                ),
            }

            message = String()

            message.data = json.dumps(
                payload,
                separators=(
                    ",",
                    ":",
                ),
            )

            self.publisher.publish(
                message
            )

            self.get_logger().info(
                "Task-aware viewpoint selected: "
                f"position={selected_position}, "
                f"uncertainty="
                f"{result.task_uncertainty:.6f} m"
            )

        except Exception as exc:
            self.get_logger().error(
                "Task-aware perception cycle failed: "
                f"{exc}"
            )


def main(args=None) -> None:
    """Run the ROS 2 task-aware perception node."""

    rclpy.init(
        args=args
    )

    node = (
        TaskAwarePerceptionNode()
    )

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()