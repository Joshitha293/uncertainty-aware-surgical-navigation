"""ROS 2 safety gate for surgical navigation commands."""

from __future__ import annotations

import json
import math
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


MAXIMUM_UNCERTAINTY_M = 0.030
MINIMUM_CLEARANCE_M = 0.005


def validate_navigation_command(
    command: dict[str, Any],
) -> tuple[bool, str]:
    """Validate a navigation command before acceptance."""

    if command.get("command") != "MOVE_CAMERA_TO_VIEWPOINT":
        return False, "Unsupported navigation command."

    if "position" not in command:
        return False, "Missing position."

    if "task_uncertainty_m" not in command:
        return False, "Missing task uncertainty."

    position = command["position"]

    if not isinstance(position, list):
        return False, "Position must be a list."

    if len(position) != 3:
        return False, "Position must contain three values."

    try:
        position = [
            float(value)
            for value in position
        ]
    except (TypeError, ValueError):
        return False, "Position contains invalid values."

    if not all(
        math.isfinite(value)
        for value in position
    ):
        return False, "Position contains non-finite values."

    try:
        uncertainty = float(
            command["task_uncertainty_m"]
        )
    except (TypeError, ValueError):
        return False, "Uncertainty is invalid."

    if not math.isfinite(
        uncertainty
    ):
        return False, "Uncertainty is non-finite."

    if uncertainty > MAXIMUM_UNCERTAINTY_M:
        return (
            False,
            (
                "Perception uncertainty exceeds "
                f"{MAXIMUM_UNCERTAINTY_M:.3f} m."
            ),
        )

    return True, "Navigation command is safe."


class SafetyGateNode(Node):
    """Validate navigation commands before acceptance."""

    def __init__(self) -> None:
        super().__init__(
            "safety_gate_node"
        )

        self.subscription = self.create_subscription(
            String,
            "navigation/viewpoint_command",
            self.command_callback,
            10,
        )

        self.safe_publisher = self.create_publisher(
            String,
            "navigation/safe_viewpoint",
            10,
        )

        self.rejected_publisher = self.create_publisher(
            String,
            "navigation/rejected_viewpoint",
            10,
        )

        self.get_logger().info(
            "Safety gate node started."
        )

    def command_callback(
        self,
        message: String,
    ) -> None:
        """Validate one navigation command."""

        try:
            command = json.loads(
                message.data
            )

            is_safe, reason = (
                validate_navigation_command(
                    command
                )
            )

            if not is_safe:
                self._reject(
                    command,
                    reason,
                )
                return

            command["safety_status"] = "SAFE"
            command["minimum_clearance_m"] = (
                MINIMUM_CLEARANCE_M
            )

            output = String()

            output.data = json.dumps(
                command,
                separators=(",", ":"),
            )

            self.safe_publisher.publish(
                output
            )

            self.get_logger().info(
                "Navigation command ACCEPTED: "
                f"position={command['position']}, "
                f"uncertainty="
                f"{float(command['task_uncertainty_m']):.6f} m"
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            self._reject(
                {},
                f"Invalid navigation command: {exc}",
            )

    def _reject(
        self,
        command: dict[str, Any],
        reason: str,
    ) -> None:
        """Reject an unsafe navigation command."""

        rejected = dict(command)

        rejected["safety_status"] = "REJECTED"
        rejected["rejection_reason"] = reason

        output = String()

        output.data = json.dumps(
            rejected,
            separators=(",", ":"),
        )

        self.rejected_publisher.publish(
            output
        )

        self.get_logger().warning(
            f"Navigation command REJECTED: {reason}"
        )


def main(args=None) -> None:
    """Run the ROS 2 safety gate."""

    rclpy.init(
        args=args
    )

    node = SafetyGateNode()

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