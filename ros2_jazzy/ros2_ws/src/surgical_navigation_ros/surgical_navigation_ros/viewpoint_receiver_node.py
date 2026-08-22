"""ROS 2 navigation command layer for task-aware viewpoint selection."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ViewpointReceiverNode(Node):
    """Receive task-aware viewpoints and publish navigation commands."""

    def __init__(self) -> None:
        super().__init__(
            "viewpoint_receiver_node"
        )

        self.subscription = self.create_subscription(
            String,
            "perception/task_aware_selection",
            self.selection_callback,
            10,
        )

        self.command_publisher = (
            self.create_publisher(
                String,
                "navigation/viewpoint_command",
                10,
            )
        )

        self.get_logger().info(
            "Viewpoint receiver and navigation "
            "command node started."
        )

    def selection_callback(
        self,
        message: String,
    ) -> None:
        """Receive, validate and forward a viewpoint command."""

        try:
            data = json.loads(
                message.data
            )

            position = data[
                "selected_position"
            ]

            candidate_count = int(
                data[
                    "candidate_count"
                ]
            )

            task_relevance = float(
                data[
                    "task_relevance"
                ]
            )

            task_alignment = float(
                data[
                    "task_alignment"
                ]
            )

            task_uncertainty = float(
                data[
                    "task_uncertainty_m"
                ]
            )

            task_score = float(
                data[
                    "task_aware_score"
                ]
            )

            generic_score = float(
                data[
                    "generic_score"
                ]
            )

            if len(position) != 3:
                raise ValueError(
                    "selected_position must contain "
                    "exactly three values."
                )

            command = {
                "command": (
                    "MOVE_CAMERA_TO_VIEWPOINT"
                ),
                "position": [
                    float(value)
                    for value in position
                ],
                "candidate_count": (
                    candidate_count
                ),
                "task_relevance": (
                    task_relevance
                ),
                "task_alignment": (
                    task_alignment
                ),
                "task_uncertainty_m": (
                    task_uncertainty
                ),
                "task_aware_score": (
                    task_score
                ),
                "generic_score": (
                    generic_score
                ),
            }

            command_message = String()

            command_message.data = json.dumps(
                command,
                separators=(
                    ",",
                    ":",
                ),
            )

            self.command_publisher.publish(
                command_message
            )

            self.get_logger().info(
                "Navigation command published: "
                f"MOVE_CAMERA_TO_VIEWPOINT "
                f"position={position}, "
                f"uncertainty="
                f"{task_uncertainty:.6f} m"
            )

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().error(
                "Invalid viewpoint selection: "
                f"{exc}"
            )


def main(args=None) -> None:
    """Run the ROS 2 viewpoint receiver."""

    rclpy.init(
        args=args
    )

    node = ViewpointReceiverNode()

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