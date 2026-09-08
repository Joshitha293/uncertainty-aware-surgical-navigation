"""Publish a visual research demo scene for the surgical navigation system."""

from __future__ import annotations

import importlib

import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from surgical_navigation_interfaces.msg import EstimatedStructure
from surgical_navigation_interfaces.msg import SafetyStatus
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


_nav_exec_module = importlib.import_module(
    'surgical_navigation_ros.navigation_execution_node'
)

make_phase2_structures = _nav_exec_module.make_phase2_structures


MARKER_TOPIC = '/navigation/demo_markers'


class DemoSceneNode(Node):
    """Publish visual markers explaining the research scene."""

    def __init__(self) -> None:
        """Create the demo scene publisher."""
        super().__init__('demo_scene_visualiser')

        self.declare_parameter('target_x', 0.12)
        self.declare_parameter('target_y', -0.03)
        self.declare_parameter('target_z', 0.02)

        self.declare_parameter('body_centre_x', 0.10)
        self.declare_parameter('body_centre_y', 0.00)
        self.declare_parameter('body_centre_z', 0.00)

        self.declare_parameter('body_size_x', 0.18)
        self.declare_parameter('body_size_y', 0.12)
        self.declare_parameter('body_size_z', 0.08)

        self._target = (
            float(self.get_parameter('target_x').value),
            float(self.get_parameter('target_y').value),
            float(self.get_parameter('target_z').value),
        )

        self._body_centre = (
            float(self.get_parameter('body_centre_x').value),
            float(self.get_parameter('body_centre_y').value),
            float(self.get_parameter('body_centre_z').value),
        )

        self._body_size = (
            float(self.get_parameter('body_size_x').value),
            float(self.get_parameter('body_size_y').value),
            float(self.get_parameter('body_size_z').value),
        )

        self._latest_estimates: dict[str, EstimatedStructure] = {}
        self._latest_status: SafetyStatus | None = None

        self._true_structures = make_phase2_structures()

        self.create_subscription(
            EstimatedStructure,
            '/navigation/estimated_structure',
            self._estimated_structure_callback,
            10,
        )

        self.create_subscription(
            SafetyStatus,
            '/navigation/safety_status',
            self._status_callback,
            10,
        )

        self._publisher = self.create_publisher(
            MarkerArray,
            MARKER_TOPIC,
            10,
        )

        self.create_timer(
            0.10,
            self._publish_markers,
        )

        self.get_logger().info(
            'Demo scene visualiser started.'
        )

    def _estimated_structure_callback(
        self,
        message: EstimatedStructure,
    ) -> None:
        """Store the latest estimate per structure."""
        self._latest_estimates[message.structure_id] = message

    def _status_callback(
        self,
        message: SafetyStatus,
    ) -> None:
        """Store the latest safety status."""
        self._latest_status = message

    @staticmethod
    def _colour(
        r: float,
        g: float,
        b: float,
        a: float,
    ) -> ColorRGBA:
        """Build a marker colour."""
        colour = ColorRGBA()
        colour.r = float(r)
        colour.g = float(g)
        colour.b = float(b)
        colour.a = float(a)
        return colour

    def _base_marker(
        self,
        marker_id: int,
        marker_type: int,
        namespace: str,
    ) -> Marker:
        """Create a basic marker."""
        marker = Marker()
        marker.header.frame_id = 'robot_base'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _body_marker(self) -> Marker:
        """Create the simplified body/workspace marker."""
        marker = self._base_marker(
            1,
            Marker.CUBE,
            'body',
        )

        marker.pose.position.x = self._body_centre[0]
        marker.pose.position.y = self._body_centre[1]
        marker.pose.position.z = self._body_centre[2]

        marker.scale.x = self._body_size[0]
        marker.scale.y = self._body_size[1]
        marker.scale.z = self._body_size[2]

        marker.color = self._colour(
            0.85, 0.75, 0.75, 0.18
        )

        return marker

    def _rcm_marker(self) -> Marker:
        """Create the RCM / insertion origin marker."""
        marker = self._base_marker(
            2,
            Marker.SPHERE,
            'rcm',
        )

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.scale.x = 0.010
        marker.scale.y = 0.010
        marker.scale.z = 0.010

        marker.color = self._colour(
            0.1, 0.4, 1.0, 1.0
        )

        return marker

    def _target_marker(self) -> Marker:
        """Create the navigation target marker."""
        marker = self._base_marker(
            3,
            Marker.SPHERE,
            'target',
        )

        marker.pose.position.x = self._target[0]
        marker.pose.position.y = self._target[1]
        marker.pose.position.z = self._target[2]

        marker.scale.x = 0.012
        marker.scale.y = 0.012
        marker.scale.z = 0.012

        marker.color = self._colour(
            0.1, 0.9, 0.2, 1.0
        )

        return marker

    def _target_text_marker(self) -> Marker:
        """Create the target label."""
        marker = self._base_marker(
            4,
            Marker.TEXT_VIEW_FACING,
            'labels',
        )

        marker.pose.position.x = self._target[0]
        marker.pose.position.y = self._target[1]
        marker.pose.position.z = self._target[2] + 0.02

        marker.scale.z = 0.012
        marker.text = 'Target'
        marker.color = self._colour(
            0.1, 0.9, 0.2, 1.0
        )

        return marker

    def _status_text_marker(self) -> Marker:
        """Create a prominent live autonomous-safety label."""
        marker = self._base_marker(
            5,
            Marker.TEXT_VIEW_FACING,
            'status',
        )

        marker.pose.position.x = 0.10
        marker.pose.position.y = -0.07
        marker.pose.position.z = 0.09

        marker.scale.z = 0.022

        if self._latest_status is None:
            marker.text = 'WAITING'

            marker.color = self._colour(
                1.0,
                1.0,
                1.0,
                1.0,
            )

            return marker

        state = str(
            self._latest_status.state
        ).lower()

        severity = str(
            self._latest_status.highest_severity
        ).lower()

        hazard = str(
            self._latest_status.active_hazard
        )

        if (
            severity == 'critical'
            or state == 'stopped'
        ):
            label = 'STOP'

            marker.color = self._colour(
                1.0,
                0.05,
                0.05,
                1.0,
            )

        elif state.startswith(
            'reacquir'
        ):
            label = 'REACQUIRE'

            marker.color = self._colour(
                1.0,
                0.60,
                0.0,
                1.0,
            )

        elif state == 'replanning':
            label = 'REPLAN'

            marker.color = self._colour(
                1.0,
                0.60,
                0.0,
                1.0,
            )

        elif state == 'recovering':
            label = 'RECOVER'

            marker.color = self._colour(
                1.0,
                0.60,
                0.0,
                1.0,
            )

        elif severity == 'recoverable':
            label = 'CAUTION'

            marker.color = self._colour(
                1.0,
                0.60,
                0.0,
                1.0,
            )

        else:
            label = 'SAFE'

            marker.color = self._colour(
                0.05,
                1.0,
                0.15,
                1.0,
            )

        if hazard not in (
            '',
            'none',
        ):
            marker.text = (
                f'{label} | {hazard}'
            )
        else:
            marker.text = label

        return marker

    def _structure_markers(self) -> list[Marker]:
        """Create markers for protected structures and uncertainty."""
        markers: list[Marker] = []

        estimates = sorted(
            self._latest_estimates.items(),
            key=lambda item: item[0],
        )

        for index, (_, estimate) in enumerate(estimates):
            x = estimate.position.x
            y = estimate.position.y
            z = estimate.position.z

            physical_radius = float(estimate.physical_radius)
            safety_margin = float(estimate.safety_margin)
            sigma = float(estimate.maximum_principal_sigma)

            body_marker = self._base_marker(
                100 + index,
                Marker.SPHERE,
                'protected_structure',
            )
            body_marker.pose.position.x = x
            body_marker.pose.position.y = y
            body_marker.pose.position.z = z
            body_marker.scale.x = physical_radius * 2.0
            body_marker.scale.y = physical_radius * 2.0
            body_marker.scale.z = physical_radius * 2.0
            body_marker.color = self._colour(
                1.0, 0.0, 0.0, 0.75
            )
            markers.append(body_marker)

            safety_shell = self._base_marker(
                200 + index,
                Marker.SPHERE,
                'safety_shell',
            )
            safety_shell.pose.position.x = x
            safety_shell.pose.position.y = y
            safety_shell.pose.position.z = z
            safety_radius = physical_radius + safety_margin
            safety_shell.scale.x = safety_radius * 2.0
            safety_shell.scale.y = safety_radius * 2.0
            safety_shell.scale.z = safety_radius * 2.0
            safety_shell.color = self._colour(
                1.0, 0.75, 0.0, 0.18
            )
            markers.append(safety_shell)

            uncertainty_shell = self._base_marker(
                300 + index,
                Marker.SPHERE,
                'uncertainty_shell',
            )
            uncertainty_shell.pose.position.x = x
            uncertainty_shell.pose.position.y = y
            uncertainty_shell.pose.position.z = z
            uncertainty_radius = physical_radius + safety_margin + (3.0 * sigma)
            uncertainty_shell.scale.x = uncertainty_radius * 2.0
            uncertainty_shell.scale.y = uncertainty_radius * 2.0
            uncertainty_shell.scale.z = uncertainty_radius * 2.0
            uncertainty_shell.color = self._colour(
                1.0, 0.45, 0.0, 0.10
            )
            markers.append(uncertainty_shell)

            label = self._base_marker(
                400 + index,
                Marker.TEXT_VIEW_FACING,
                'structure_labels',
            )
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = z + uncertainty_radius + 0.01
            label.scale.z = 0.010
            label.text = f'Protected anatomy {index}'
            label.color = self._colour(
                1.0, 1.0, 1.0, 1.0
            )
            markers.append(label)

        if not estimates:
            for index, structure in enumerate(self._true_structures):
                placeholder = self._base_marker(
                    500 + index,
                    Marker.SPHERE,
                    'true_structure_placeholder',
                )
                placeholder.pose.position.x = float(structure.centre[0])
                placeholder.pose.position.y = float(structure.centre[1])
                placeholder.pose.position.z = float(structure.centre[2])
                placeholder.scale.x = float(structure.physical_radius * 2.0)
                placeholder.scale.y = float(structure.physical_radius * 2.0)
                placeholder.scale.z = float(structure.physical_radius * 2.0)
                placeholder.color = self._colour(
                    0.7, 0.0, 0.0, 0.25
                )
                markers.append(placeholder)

        return markers

    def _publish_markers(self) -> None:
        """Publish the full demo marker set."""
        marker_array = MarkerArray()
        marker_array.markers.append(self._body_marker())
        marker_array.markers.append(self._rcm_marker())
        marker_array.markers.append(self._target_marker())
        marker_array.markers.append(self._target_text_marker())
        marker_array.markers.append(self._status_text_marker())
        marker_array.markers.extend(self._structure_markers())

        self._publisher.publish(marker_array)


def main(args=None) -> None:
    """Run the demo scene visualiser."""
    rclpy.init(args=args)

    node = DemoSceneNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
