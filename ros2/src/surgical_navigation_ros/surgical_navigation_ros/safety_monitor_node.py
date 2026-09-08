"""Expose the Phase 7 runtime safety monitors through ROS 2."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import UInt32

from surgical_navigation_interfaces.msg import (
    EstimatedStructure,
    SafetyStatus,
)

from surgical_navigation_ros.core_adapter import (
    build_runtime_snapshot,
    evaluate_runtime_safety,
    highest_severity_event,
    ordered_joint_positions,
    recommended_state_for_event,
    RuntimeSafetyConfig,
)


class RuntimeSafetyMonitorNode(Node):
    """Bridge ROS runtime state into the existing Phase 7 safety logic."""

    def __init__(self) -> None:
        """Create the runtime-safety ROS node."""
        super().__init__(
            'runtime_safety_monitor'
        )

        self.declare_parameter(
            'monitor_period_s',
            0.1,
        )

        self.declare_parameter(
            'max_perception_age_steps',
            3,
        )

        self.declare_parameter(
            'uncertainty_reacquire_sigma',
            0.015,
        )

        self.declare_parameter(
            'uncertainty_stop_sigma',
            0.030,
        )

        self.declare_parameter(
            'clearance_recovery_threshold',
            0.002,
        )

        self.declare_parameter(
            'clearance_stop_threshold',
            0.0,
        )

        self.declare_parameter(
            'joint_limit_warning_margin',
            0.05,
        )

        self.declare_parameter(
            'tracking_error_recovery_threshold',
            0.005,
        )

        self.declare_parameter(
            'tracking_error_stop_threshold',
            0.015,
        )

        self.declare_parameter(
            'maximum_execution_steps',
            60,
        )

        self._config = RuntimeSafetyConfig(
            max_perception_age_steps=int(
                self.get_parameter(
                    'max_perception_age_steps'
                ).value
            ),
            uncertainty_reacquire_sigma=float(
                self.get_parameter(
                    'uncertainty_reacquire_sigma'
                ).value
            ),
            uncertainty_stop_sigma=float(
                self.get_parameter(
                    'uncertainty_stop_sigma'
                ).value
            ),
            clearance_recovery_threshold=float(
                self.get_parameter(
                    'clearance_recovery_threshold'
                ).value
            ),
            clearance_stop_threshold=float(
                self.get_parameter(
                    'clearance_stop_threshold'
                ).value
            ),
            joint_limit_warning_margin=float(
                self.get_parameter(
                    'joint_limit_warning_margin'
                ).value
            ),
            tracking_error_recovery_threshold=float(
                self.get_parameter(
                    'tracking_error_recovery_threshold'
                ).value
            ),
            tracking_error_stop_threshold=float(
                self.get_parameter(
                    'tracking_error_stop_threshold'
                ).value
            ),
            maximum_execution_steps=int(
                self.get_parameter(
                    'maximum_execution_steps'
                ).value
            ),
        )

        self._step_index = 0

        self._latest_perception_step: int | None = None

        self._maximum_principal_sigma: (
            float | None
        ) = None

        self._joint_positions: (
            np.ndarray | None
        ) = None

        self._predicted_clearance: (
            float | None
        ) = None

        self._tracking_error: (
            float | None
        ) = None

        self._execution_steps: (
            int | None
        ) = None

        self._status_publisher = (
            self.create_publisher(
                SafetyStatus,
                '/navigation/safety_status',
                10,
            )
        )

        self.create_subscription(
            EstimatedStructure,
            '/navigation/estimated_structure',
            self._on_estimated_structure,
            10,
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_state,
            10,
        )

        self.create_subscription(
            Float64,
            '/navigation/predicted_clearance',
            self._on_predicted_clearance,
            10,
        )

        self.create_subscription(
            Float64,
            '/navigation/tracking_error',
            self._on_tracking_error,
            10,
        )

        self.create_subscription(
            UInt32,
            '/navigation/execution_steps',
            self._on_execution_steps,
            10,
        )

        monitor_period = float(
            self.get_parameter(
                'monitor_period_s'
            ).value
        )

        if (
            not np.isfinite(monitor_period)
            or monitor_period <= 0.0
        ):
            raise ValueError(
                'monitor_period_s must be finite and positive.'
            )

        self.create_timer(
            monitor_period,
            self._monitor_runtime_state,
        )

        self.get_logger().info(
            'Phase 7 runtime safety monitor bridge started.'
        )

    def _on_estimated_structure(
        self,
        message: EstimatedStructure,
    ) -> None:
        """Store the latest perception uncertainty."""
        self._latest_perception_step = (
            self._step_index
        )

        if not message.valid:
            self._maximum_principal_sigma = (
                float('nan')
            )
            return

        self._maximum_principal_sigma = float(
            message.maximum_principal_sigma
        )

    def _on_joint_state(
        self,
        message: JointState,
    ) -> None:
        """Store current robot joint positions."""
        try:
            self._joint_positions = (
                ordered_joint_positions(
                    message.name,
                    message.position,
                )
            )
        except ValueError as error:
            self._joint_positions = None

            self.get_logger().error(
                str(error)
            )

    def _on_predicted_clearance(
        self,
        message: Float64,
    ) -> None:
        """Store the planner's predicted protected-region clearance."""
        self._predicted_clearance = float(
            message.data
        )

    def _on_tracking_error(
        self,
        message: Float64,
    ) -> None:
        """Store current trajectory tracking error."""
        self._tracking_error = float(
            message.data
        )

    def _on_execution_steps(
        self,
        message: UInt32,
    ) -> None:
        """Store the execution-step count."""
        self._execution_steps = int(
            message.data
        )

    def _missing_inputs(
        self,
    ) -> list[str]:
        """Return runtime inputs that have not arrived yet."""
        missing: list[str] = []

        if self._latest_perception_step is None:
            missing.append(
                'estimated_structure'
            )

        if self._maximum_principal_sigma is None:
            missing.append(
                'perception_uncertainty'
            )

        if self._joint_positions is None:
            missing.append(
                'joint_states'
            )

        if self._predicted_clearance is None:
            missing.append(
                'predicted_clearance'
            )

        if self._tracking_error is None:
            missing.append(
                'tracking_error'
            )

        if self._execution_steps is None:
            missing.append(
                'execution_steps'
            )

        return missing

    def _base_status(
        self,
    ) -> SafetyStatus:
        """Create common metadata for a safety-status message."""
        message = SafetyStatus()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )

        message.header.frame_id = (
            'robot_base'
        )

        message.replan_count = 0
        message.reacquisition_count = 0
        message.recovery_count = 0
        message.stop_count = 0
        message.failure_count = 0

        return message

    def _publish_waiting_status(
        self,
        missing: list[str],
    ) -> None:
        """Publish status while required runtime inputs are unavailable."""
        message = self._base_status()

        message.state = 'initialising'
        message.highest_severity = 'advisory'
        message.active_hazard = 'none'

        message.message = (
            'Waiting for runtime inputs: '
            + ', '.join(missing)
        )

        self._status_publisher.publish(
            message
        )

    def _monitor_runtime_state(
        self,
    ) -> None:
        """Evaluate one Phase 7 safety-monitoring step."""
        self._step_index += 1

        missing = self._missing_inputs()

        if missing:
            self._publish_waiting_status(
                missing
            )
            return

        snapshot = build_runtime_snapshot(
            step_index=self._step_index,
            latest_perception_step=(
                self._latest_perception_step
            ),
            maximum_principal_sigma=(
                self._maximum_principal_sigma
            ),
            predicted_clearance=(
                self._predicted_clearance
            ),
            joint_positions=(
                self._joint_positions
            ),
            tracking_error=(
                self._tracking_error
            ),
            execution_steps=(
                self._execution_steps
            ),
        )

        events = evaluate_runtime_safety(
            snapshot,
            self._config,
        )

        highest_event = (
            highest_severity_event(
                events
            )
        )

        message = self._base_status()

        if highest_event is None:
            message.state = 'executing'
            message.highest_severity = 'none'
            message.active_hazard = 'none'
            message.message = (
                'No active runtime safety hazards.'
            )

        else:
            recommended_state = (
                recommended_state_for_event(
                    highest_event
                )
            )

            if recommended_state is None:
                message.state = 'executing'
            else:
                message.state = (
                    recommended_state.value
                )

            message.highest_severity = (
                highest_event.severity.value
            )

            message.active_hazard = (
                highest_event.hazard.value
            )

            message.message = (
                highest_event.message
            )

        self._status_publisher.publish(
            message
        )


def main(
    args: list[str] | None = None,
) -> None:
    """Run the ROS 2 runtime safety-monitor node."""
    rclpy.init(
        args=args
    )

    node = RuntimeSafetyMonitorNode()

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
