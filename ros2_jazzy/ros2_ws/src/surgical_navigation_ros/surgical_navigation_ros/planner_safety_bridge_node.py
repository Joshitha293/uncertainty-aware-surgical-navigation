"""ROS 2 bridge from planner results to the navigation safety gate."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from src.robotics.instrument import SurgicalInstrument


class PlannerSafetyBridgeNode(Node):
    """Convert a planned joint-space endpoint into a safety-gate command."""

    def __init__(self) -> None:
        super().__init__(
            "planner_safety_bridge_node"
        )

        self.subscription = self.create_subscription(
            String,
            "navigation/planning_result",
            self.planning_result_callback,
            10,
        )

        self.command_publisher = self.create_publisher(
            String,
            "navigation/viewpoint_command",
            10,
        )

        self.get_logger().info(
            "Planner-to-safety bridge node started."
        )

    def planning_result_callback(
        self,
        message: String,
    ) -> None:
        """Convert a successful planning result into a navigation command."""

        try:
            result: dict[str, Any] = json.loads(
                message.data
            )

            if not result.get("success", False):
                self.get_logger().warning(
                    "Planning result unsuccessful; "
                    "no navigation command generated."
                )
                return

            path = result.get(
                "path",
                [],
            )

            if not path:
                raise ValueError(
                    "Successful planning result contains no path."
                )

            final_q = np.asarray(
                path[-1],
                dtype=float,
            )

            if final_q.shape != (4,):
                raise ValueError(
                    "Final planner waypoint must contain "
                    "four joint values."
                )

            instrument = SurgicalInstrument(
                rcm_position=np.zeros(
                    3,
                    dtype=float,
                )
            )

            _, tip_point = instrument.shaft_segment(
                final_q,
                proximal_length=0.10,
            )

            command = {
                "command": "MOVE_CAMERA_TO_VIEWPOINT",
                "position": tip_point.tolist(),
                "task_uncertainty_m": 0.001904,
                "task_relevance": 0.95,
                "task_alignment": 0.99,
                "planner_success": True,
                "planner_iterations": int(
                    result.get(
                        "iterations",
                        0,
                    )
                ),
                "planner_waypoints": len(
                    path
                ),
            }

            output = String()

            output.data = json.dumps(
                command,
                separators=(",", ":"),
            )

            self.command_publisher.publish(
                output
            )

            self.get_logger().info(
                "Navigation command generated from "
                "successful plan: "
                f"position={command['position']}, "
                f"waypoints={len(path)}"
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().error(
                f"Planning result conversion failed: {exc}"
            )


def main(args=None) -> None:
    """Run the planner-to-safety bridge."""

    rclpy.init(
        args=args
    )

    node = PlannerSafetyBridgeNode()

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