"""Publish simulated uncertain anatomy estimates through ROS 2."""

from __future__ import annotations

import importlib

import numpy as np
import rclpy
from rclpy.node import Node

from surgical_navigation_interfaces.msg import EstimatedStructure

from surgical_navigation_ros.navigation_execution_node import (
    make_phase2_structures,
)


_uncertainty_module = importlib.import_module(
    'src.perception.uncertainty'
)

PositionUncertainty = (
    _uncertainty_module.PositionUncertainty
)

make_estimated_structure = (
    _uncertainty_module.make_estimated_structure
)


ESTIMATED_STRUCTURE_TOPIC = (
    '/navigation/estimated_structure'
)


class SimulatedPerceptionBridgeNode(Node):
    """Publish uncertainty-aware simulated anatomical observations."""

    def __init__(self) -> None:
        """Create the simulated perception bridge."""
        super().__init__(
            'simulated_perception_bridge'
        )

        self.declare_parameter(
            'publish_period_s',
            0.10,
        )

        self.declare_parameter(
            'position_sigma_m',
            0.003,
        )

        self.declare_parameter(
            'random_seed',
            17,
        )

        publish_period = float(
            self.get_parameter(
                'publish_period_s'
            ).value
        )

        sigma = float(
            self.get_parameter(
                'position_sigma_m'
            ).value
        )

        seed = int(
            self.get_parameter(
                'random_seed'
            ).value
        )

        if (
            not np.isfinite(publish_period)
            or publish_period <= 0.0
        ):
            raise ValueError(
                'publish_period_s must be finite and positive.'
            )

        if (
            not np.isfinite(sigma)
            or sigma < 0.0
        ):
            raise ValueError(
                'position_sigma_m must be finite and non-negative.'
            )

        self._structures = (
            make_phase2_structures()
        )

        self._uncertainty = (
            PositionUncertainty.isotropic(
                sigma
            )
        )

        self._rng = np.random.default_rng(
            seed
        )

        self._publisher = (
            self.create_publisher(
                EstimatedStructure,
                ESTIMATED_STRUCTURE_TOPIC,
                10,
            )
        )

        self.create_timer(
            publish_period,
            self._publish_estimates,
        )

        self.get_logger().info(
            'Simulated uncertainty-aware perception bridge started.'
        )

        self.get_logger().info(
            'Perception position sigma: '
            f'{sigma:.6f} m.'
        )

    def _publish_estimates(
        self,
    ) -> None:
        """Publish one uncertain observation for every protected structure."""
        for index, structure in enumerate(
            self._structures
        ):
            estimate = (
                make_estimated_structure(
                    true_structure=structure,
                    uncertainty=(
                        self._uncertainty
                    ),
                    rng=self._rng,
                )
            )

            message = EstimatedStructure()

            message.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            message.header.frame_id = (
                'robot_base'
            )

            message.structure_id = (
                f'protected_structure_{index}'
            )

            message.position.x = float(
                estimate.estimated_centre[0]
            )

            message.position.y = float(
                estimate.estimated_centre[1]
            )

            message.position.z = float(
                estimate.estimated_centre[2]
            )

            covariance = (
                estimate
                .uncertainty
                .covariance
                .reshape(-1)
            )

            message.covariance = [
                float(value)
                for value in covariance
            ]

            message.physical_radius = float(
                estimate.physical_radius
            )

            message.safety_margin = float(
                estimate.base_safety_margin
            )

            message.maximum_principal_sigma = float(
                estimate
                .uncertainty
                .principal_sigma
            )

            message.valid = True

            self._publisher.publish(
                message
            )


def main(
    args=None,
) -> None:
    """Run the simulated perception bridge."""
    rclpy.init(
        args=args
    )

    node = SimulatedPerceptionBridgeNode()

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


if __name__ == '__main__':
    main()
