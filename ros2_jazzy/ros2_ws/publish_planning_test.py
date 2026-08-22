import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PlanningTestPublisher(Node):

    def __init__(self):
        super().__init__("planning_test_publisher")

        self.publisher = self.create_publisher(
            String,
            "/navigation/planning_request",
            10,
        )

        self.timer = self.create_timer(
            1.0,
            self.publish_request,
        )

        self.sent = False

    def publish_request(self):
        if self.sent:
            return

        request = {
            "start_q": [
                0.0,
                0.0,
                0.10,
                0.0,
            ],
            "goal_q": [
                0.6108652382,
                0.4363323120,
                0.25,
                0.0,
            ],
            "instrument_position": [
                0.0,
                0.0,
                0.0,
            ],
            "structures": [
                {
                    "centre": [
                        0.14,
                        0.04,
                        0.0,
                    ],
                    "physical_radius": 0.025,
                    "safety_margin": 0.015,
                },
                {
                    "centre": [
                        0.18,
                        -0.06,
                        0.02,
                    ],
                    "physical_radius": 0.025,
                    "safety_margin": 0.015,
                },
            ],
            "instrument_radius": 0.006,
            "proximal_length": 0.10,
            "max_iterations": 5000,
            "step_size": 0.08,
            "goal_bias": 0.15,
            "edge_resolution": 20,
            "seed": 20000,
        }

        message = String()
        message.data = json.dumps(
            request,
            separators=(",", ":"),
        )

        self.publisher.publish(message)

        self.get_logger().info(
            "Collision-aware planning request published:"
        )
        self.get_logger().info(
            message.data
        )

        self.sent = True


def main():
    rclpy.init()

    node = PlanningTestPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()