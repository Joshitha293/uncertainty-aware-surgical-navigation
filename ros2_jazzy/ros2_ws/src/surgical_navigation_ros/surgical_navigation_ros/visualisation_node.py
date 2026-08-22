"""RViz visualisation for the uncertainty-aware surgical navigation system."""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class SurgicalNavigationVisualisationNode(Node):
    """Visualise the actual ROS planning result in RViz."""

    def __init__(self) -> None:
        super().__init__(
            "surgical_navigation_visualisation"
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            "/surgical_navigation/markers",
            10,
        )

        self.status_publisher = self.create_publisher(
            String,
            "/surgical_navigation/status",
            10,
        )

        self.planning_subscription = (
            self.create_subscription(
                String,
                "/navigation/planning_result",
                self.planning_result_callback,
                10,
            )
        )

        # Same RCM and workspace used by the validated
        # planning test request.
        self.rcm_position = np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        )

        self.structures = [
            {
                "centre": np.array(
                    [0.14, 0.04, 0.0],
                    dtype=float,
                ),
                "physical_radius": 0.025,
                "safety_margin": 0.015,
            },
            {
                "centre": np.array(
                    [0.18, -0.06, 0.02],
                    dtype=float,
                ),
                "physical_radius": 0.025,
                "safety_margin": 0.015,
            },
        ]

        self.instrument_radius = 0.006
        self.proximal_length = 0.10

        self.latest_path = None
        self.latest_iterations = 0
        self.latest_success = False

        self.get_logger().info(
            "Surgical navigation visualisation node started."
        )

        self.get_logger().info(
            "Waiting for /navigation/planning_result..."
        )

    @staticmethod
    def shaft_direction(
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        """Return the instrument shaft unit direction."""

        return np.array(
            [
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=float,
        )

    def tip_position(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Calculate instrument tip position from [yaw, pitch, insertion, roll]."""

        yaw = float(q[0])
        pitch = float(q[1])
        insertion = float(q[2])

        direction = self.shaft_direction(
            yaw,
            pitch,
        )

        return (
            self.rcm_position
            + insertion * direction
        )

    def shaft_segment(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return proximal shaft point and tool tip."""

        yaw = float(q[0])
        pitch = float(q[1])

        direction = self.shaft_direction(
            yaw,
            pitch,
        )

        proximal_point = (
            self.rcm_position
            - self.proximal_length * direction
        )

        tip_point = self.tip_position(q)

        return proximal_point, tip_point

    @staticmethod
    def make_point(
        position: np.ndarray,
    ) -> Point:
        """Convert NumPy coordinates to a ROS Point."""

        point = Point()

        point.x = float(position[0])
        point.y = float(position[1])
        point.z = float(position[2])

        return point

    @staticmethod
    def set_colour(
        marker: Marker,
        red: float,
        green: float,
        blue: float,
        alpha: float,
    ) -> None:
        """Set marker colour."""

        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    def create_sphere_marker(
        self,
        marker_id: int,
        namespace: str,
        position: np.ndarray,
        radius: float,
        red: float,
        green: float,
        blue: float,
        alpha: float,
    ) -> Marker:
        """Create a spherical RViz marker."""

        marker = Marker()

        marker.header.frame_id = "world"
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(
            position[0]
        )
        marker.pose.position.y = float(
            position[1]
        )
        marker.pose.position.z = float(
            position[2]
        )

        marker.pose.orientation.w = 1.0

        marker.scale.x = 2.0 * radius
        marker.scale.y = 2.0 * radius
        marker.scale.z = 2.0 * radius

        self.set_colour(
            marker,
            red,
            green,
            blue,
            alpha,
        )

        return marker

    def planning_result_callback(
        self,
        message: String,
    ) -> None:
        """Receive and visualise the actual planner result."""

        try:
            result = json.loads(
                message.data
            )

            self.latest_success = bool(
                result.get(
                    "success",
                    False,
                )
            )

            self.latest_iterations = int(
                result.get(
                    "iterations",
                    0,
                )
            )

            path = result.get(
                "path",
                [],
            )

            if not self.latest_success:
                self.get_logger().warning(
                    "Planner returned an unsuccessful result."
                )

                self.latest_path = None

                return

            if not path:
                self.get_logger().warning(
                    "Planner reported success but returned no path."
                )

                self.latest_path = None

                return

            self.latest_path = np.asarray(
                path,
                dtype=float,
            )

            if (
                self.latest_path.ndim != 2
                or self.latest_path.shape[1] != 4
            ):
                raise ValueError(
                    "Planner path must have shape N x 4."
                )

            self.get_logger().info(
                "Received successful planning result: "
                f"iterations={self.latest_iterations}, "
                f"waypoints={len(self.latest_path)}"
            )

            self.publish_scene()

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            self.get_logger().error(
                f"Failed to visualise planning result: {exc}"
            )

    def publish_scene(self) -> None:
        """Publish the complete surgical scene to RViz."""

        markers = MarkerArray()

        # --------------------------------------------------
        # 1. Remote Centre of Motion
        # --------------------------------------------------

        rcm_marker = self.create_sphere_marker(
            marker_id=0,
            namespace="rcm",
            position=self.rcm_position,
            radius=0.008,
            red=1.0,
            green=1.0,
            blue=0.0,
            alpha=1.0,
        )

        markers.markers.append(
            rcm_marker
        )

        # --------------------------------------------------
        # 2. Anatomical structures
        # --------------------------------------------------

        for index, structure in enumerate(
            self.structures
        ):
            centre = structure["centre"]

            physical_radius = float(
                structure["physical_radius"]
            )

            safety_radius = (
                physical_radius
                + float(
                    structure["safety_margin"]
                )
            )

            # Physical anatomy.
            anatomy_marker = self.create_sphere_marker(
                marker_id=10 + index,
                namespace="anatomy",
                position=centre,
                radius=physical_radius,
                red=0.85,
                green=0.10,
                blue=0.10,
                alpha=0.90,
            )

            markers.markers.append(
                anatomy_marker
            )

            # Protected safety envelope.
            safety_marker = self.create_sphere_marker(
                marker_id=100 + index,
                namespace="safety_envelope",
                position=centre,
                radius=safety_radius,
                red=1.0,
                green=0.60,
                blue=0.05,
                alpha=0.20,
            )

            markers.markers.append(
                safety_marker
            )

        # --------------------------------------------------
        # 3. Planned trajectory
        # --------------------------------------------------

        if (
            self.latest_path is not None
            and self.latest_success
        ):
            path_marker = Marker()

            path_marker.header.frame_id = "world"
            path_marker.header.stamp = (
                self.get_clock().now().to_msg()
            )

            path_marker.ns = (
                "planned_instrument_trajectory"
            )

            path_marker.id = 200
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD

            path_marker.scale.x = 0.004

            self.set_colour(
                path_marker,
                0.0,
                0.80,
                1.0,
                1.0,
            )

            for q in self.latest_path:
                tip = self.tip_position(q)

                path_marker.points.append(
                    self.make_point(tip)
                )

            markers.markers.append(
                path_marker
            )

            # --------------------------------------------------
            # 4. Start instrument pose
            # --------------------------------------------------

            start_q = self.latest_path[0]

            proximal_point, start_tip = (
                self.shaft_segment(
                    start_q
                )
            )

            start_instrument_marker = Marker()

            start_instrument_marker.header.frame_id = (
                "world"
            )

            start_instrument_marker.header.stamp = (
                self.get_clock().now().to_msg()
            )

            start_instrument_marker.ns = (
                "instrument_start"
            )

            start_instrument_marker.id = 300

            start_instrument_marker.type = (
                Marker.LINE_STRIP
            )

            start_instrument_marker.action = (
                Marker.ADD
            )

            start_instrument_marker.scale.x = 0.008

            self.set_colour(
                start_instrument_marker,
                0.10,
                0.90,
                0.30,
                1.0,
            )

            start_instrument_marker.points.append(
                self.make_point(
                    proximal_point
                )
            )

            start_instrument_marker.points.append(
                self.make_point(
                    start_tip
                )
            )

            markers.markers.append(
                start_instrument_marker
            )

            # --------------------------------------------------
            # 5. Goal marker
            # --------------------------------------------------

            goal_q = self.latest_path[-1]

            goal_tip = self.tip_position(
                goal_q
            )

            goal_marker = self.create_sphere_marker(
                marker_id=400,
                namespace="planned_goal",
                position=goal_tip,
                radius=0.010,
                red=0.10,
                green=1.0,
                blue=0.20,
                alpha=1.0,
            )

            markers.markers.append(
                goal_marker
            )

            # --------------------------------------------------
            # 6. Waypoint markers
            # --------------------------------------------------

            for index, q in enumerate(
                self.latest_path
            ):
                tip = self.tip_position(q)

                waypoint_marker = self.create_sphere_marker(
                    marker_id=500 + index,
                    namespace="rrt_waypoints",
                    position=tip,
                    radius=0.003,
                    red=0.0,
                    green=0.60,
                    blue=1.0,
                    alpha=0.9,
                )

                markers.markers.append(
                    waypoint_marker
                )

            # --------------------------------------------------
            # 7. Status information
            # --------------------------------------------------

            status = String()

            status.data = (
                "RRT SUCCESS | "
                f"iterations={self.latest_iterations} | "
                f"waypoints={len(self.latest_path)}"
            )

            self.status_publisher.publish(
                status
            )

        self.marker_publisher.publish(
            markers
        )


def main(args=None) -> None:
    """Run the surgical-navigation visualisation node."""

    rclpy.init(
        args=args
    )

    node = SurgicalNavigationVisualisationNode()

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