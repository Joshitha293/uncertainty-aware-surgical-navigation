"""ROS 2 bridge to the existing collision-aware RRT planner."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import plan_rrt


class PlannerNode(Node):
    """Receive joint-space goals and execute the existing RRT planner."""

    def __init__(self) -> None:
        super().__init__("planner_node")

        self.subscription = self.create_subscription(
            String,
            "navigation/planning_request",
            self.planning_callback,
            10,
        )

        self.result_publisher = self.create_publisher(
            String,
            "navigation/planning_result",
            10,
        )

        self.get_logger().info(
            "Collision-aware RRT planner node started."
        )

    def planning_callback(
        self,
        message: String,
    ) -> None:
        """Process one planning request."""

        try:
            request: dict[str, Any] = json.loads(
                message.data
            )

            start_q = np.asarray(
                request["start_q"],
                dtype=float,
            )

            goal_q = np.asarray(
                request["goal_q"],
                dtype=float,
            )

            instrument_position = np.asarray(
                request.get(
                    "instrument_position",
                    [0.0, 0.0, 0.0],
                ),
                dtype=float,
            )

            instrument = SurgicalInstrument(
                rcm_position=instrument_position,
            )

            structures = tuple(
                SphericalStructure(
                    centre=np.asarray(
                        structure["centre"],
                        dtype=float,
                    ),
                    physical_radius=float(
                        structure["physical_radius"]
                    ),
                    safety_margin=float(
                        structure["safety_margin"]
                    ),
                )
                for structure in request.get(
                    "structures",
                    [],
                )
            )

            instrument_radius = float(
                request.get(
                    "instrument_radius",
                    0.003,
                )
            )

            proximal_length = float(
                request.get(
                    "proximal_length",
                    0.1,
                )
            )

            max_iterations = int(
                request.get(
                    "max_iterations",
                    5000,
                )
            )

            step_size = float(
                request.get(
                    "step_size",
                    0.08,
                )
            )

            goal_bias = float(
                request.get(
                    "goal_bias",
                    0.15,
                )
            )

            edge_resolution = int(
                request.get(
                    "edge_resolution",
                    20,
                )
            )

            seed = int(
                request.get(
                    "seed",
                    7,
                )
            )

            result = plan_rrt(
                instrument=instrument,
                start_q=start_q,
                goal_q=goal_q,
                structures=structures,
                instrument_radius=instrument_radius,
                proximal_length=proximal_length,
                max_iterations=max_iterations,
                step_size=step_size,
                goal_bias=goal_bias,
                edge_resolution=edge_resolution,
                seed=seed,
            )

            response = {
                "success": bool(
                    result.success
                ),
                "iterations": int(
                    result.iterations
                ),
                "path": (
                    result.path.tolist()
                    if result.path is not None
                    else []
                ),
            }

            output = String()

            output.data = json.dumps(
                response,
                separators=(",", ":"),
            )

            self.result_publisher.publish(
                output
            )

            self.get_logger().info(
                "Planning completed: "
                f"success={result.success}, "
                f"iterations={result.iterations}, "
                f"waypoints={len(result.path)}"
            )

        except Exception as exc:
            self.get_logger().error(
                f"Planning request failed: {exc}"
            )


def main(args=None) -> None:
    """Run the planner."""

    rclpy.init(
        args=args
    )

    node = PlannerNode()

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