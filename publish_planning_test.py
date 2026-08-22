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
                0.05,
                0.05,
                0.15,
                0.05,
            ],
            "instrument_position": [
                0.0,
                0.0,
                0.0,
            ],
            "structures": [],
            "instrument_radius": 0.003,
            "proximal_length": 0.1,
            "max_iterations": 500,
            "step_size": 0.08,
            "goal_bias": 0.2,
            "edge_resolution": 20,
            "seed": 7,
        }

        message = String()

        message.data = json.dumps(
            request,
            separators=(",", ":"),
        )

        self.publisher.publish(message)

        self.get_logger().info(
            "Planning request published:"
        )

        self.get_logger().info(
            message.data
        )

        self.sent = True

        rclpy.shutdown()


def main():
    rclpy.init()

    node = PlanningTestPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()