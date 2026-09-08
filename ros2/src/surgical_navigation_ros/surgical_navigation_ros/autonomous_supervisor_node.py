"""Bridge the Phase 7 autonomous supervisor into ROS 2."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import UInt32
from surgical_navigation_interfaces.msg import EstimatedStructure
from surgical_navigation_interfaces.msg import SafetyStatus
from surgical_navigation_ros.core_adapter import (
    AutonomousExecutionController,
    RuntimeSafetyConfig,
)
from surgical_navigation_ros.core_adapter import build_runtime_snapshot
from surgical_navigation_ros.core_adapter import highest_severity_event
from surgical_navigation_ros.core_adapter import SafetyState


COMMAND_TOPIC = '/navigation/supervisor_command'
STATUS_TOPIC = '/navigation/safety_status'


class AutonomousSupervisorLogic:
    """Wrap the existing Phase 7 autonomous execution controller."""

    def __init__(
        self,
        safety_config: RuntimeSafetyConfig | None = None,
    ) -> None:
        """Create one independent autonomous-supervision episode."""
        self._safety_config = safety_config
        self._controller = self._make_controller()
        self._started = False

    @staticmethod
    def _accepted_callback(_snapshot) -> bool:
        """Acknowledge a ROS-side response request."""
        return True

    def _make_controller(self) -> AutonomousExecutionController:
        """Create the existing Phase 7 controller."""
        return AutonomousExecutionController(
            planner_callback=self._accepted_callback,
            reacquisition_callback=self._accepted_callback,
            recovery_callback=self._accepted_callback,
            safety_config=self._safety_config,
        )

    @property
    def controller(self) -> AutonomousExecutionController:
        """Return the underlying Phase 7 controller."""
        return self._controller

    def reset(self) -> None:
        """Start a new independent execution episode."""
        self._controller = self._make_controller()
        self._started = False

    def process(self, snapshot):
        """Process one runtime snapshot through the Phase 7 controller."""
        if not self._started:
            decision = self._controller.start(
                snapshot
            )

            self._started = True

            return decision

        if self._controller.state in (
            SafetyState.STOPPED,
            SafetyState.COMPLETED,
            SafetyState.FAILED,
        ):
            return None

        return self._controller.process_snapshot(
            snapshot
        )


def command_from_decision(decision) -> str | None:
    """Map a Phase 7 decision to a ROS execution command."""
    if decision is None:
        return None

    action = decision.action.value

    if action in (
        'replan',
        'reacquire',
        'recover',
        'stop',
    ):
        return action

    if action == 'fail':
        return 'stop'

    return None


class AutonomousSupervisorNode(Node):
    """Run Phase 7 autonomous safety supervision from live ROS data."""

    def __init__(self) -> None:
        """Create ROS subscriptions and supervision publishers."""
        super().__init__(
            'autonomous_supervisor'
        )

        self.declare_parameter(
            'demo_mode',
            False,
        )

        demo_mode = bool(
            self.get_parameter(
                'demo_mode'
            ).value
        )

        safety_config = None

        if demo_mode:
            safety_config = RuntimeSafetyConfig(
                tracking_error_recovery_threshold=1.0,
                tracking_error_stop_threshold=2.0,
                joint_limit_warning_margin=0.0,
            )

            self.get_logger().info(
                'Demo safety profile enabled: '
                'tracking thresholds relaxed; '
                'uncertainty and clearance thresholds unchanged.'
            )

        self._logic = AutonomousSupervisorLogic(
            safety_config=safety_config,
        )

        self._joint_names: tuple[str, ...] | None = None
        self._joint_positions: np.ndarray | None = None

        self._maximum_principal_sigma: float | None = None
        self._predicted_clearance: float | None = None
        self._tracking_error: float | None = None

        self._execution_steps: int | None = None
        self._latest_perception_step: int | None = None

        self._perception_waiting_for_step = False
        self._last_processed_step: int | None = None

        self._command_publisher = self.create_publisher(
            String,
            COMMAND_TOPIC,
            10,
        )

        self._status_publisher = self.create_publisher(
            SafetyStatus,
            STATUS_TOPIC,
            10,
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10,
        )

        self.create_subscription(
            EstimatedStructure,
            '/navigation/estimated_structure',
            self._estimated_structure_callback,
            10,
        )

        self.create_subscription(
            Float64,
            '/navigation/predicted_clearance',
            self._clearance_callback,
            10,
        )

        self.create_subscription(
            Float64,
            '/navigation/tracking_error',
            self._tracking_error_callback,
            10,
        )

        self.create_subscription(
            UInt32,
            '/navigation/execution_steps',
            self._execution_steps_callback,
            10,
        )

        self.get_logger().info(
            'Autonomous Phase 7 ROS supervisor ready.'
        )

    def _joint_callback(
        self,
        message: JointState,
    ) -> None:
        """Store the latest joint measurement."""
        positions = np.asarray(
            message.position,
            dtype=float,
        )

        if (
            positions.ndim != 1
            or len(message.name) != positions.size
            or positions.size == 0
            or not np.all(np.isfinite(positions))
        ):
            return

        self._joint_names = tuple(
            str(name)
            for name in message.name
        )

        self._joint_positions = positions.copy()

    def _estimated_structure_callback(
        self,
        message: EstimatedStructure,
    ) -> None:
        """Store the latest positional uncertainty observation."""
        sigma = float(
            message.maximum_principal_sigma
        )

        if not np.isfinite(sigma):
            return

        self._maximum_principal_sigma = sigma

        if self._execution_steps is None:
            self._perception_waiting_for_step = True
        else:
            self._latest_perception_step = int(
                self._execution_steps
            )

            self._perception_waiting_for_step = False

    def _clearance_callback(
        self,
        message: Float64,
    ) -> None:
        """Store latest predicted protected-region clearance."""
        value = float(
            message.data
        )

        if np.isfinite(value):
            self._predicted_clearance = value

    def _tracking_error_callback(
        self,
        message: Float64,
    ) -> None:
        """Store latest trajectory-tracking error."""
        value = float(
            message.data
        )

        if np.isfinite(value):
            self._tracking_error = value

    def _execution_steps_callback(
        self,
        message: UInt32,
    ) -> None:
        """Process one new execution step."""
        step = int(
            message.data
        )

        if (
            self._last_processed_step is not None
            and step < self._last_processed_step
        ):
            self.get_logger().info(
                'Execution-step reset detected; '
                'starting new supervision episode.'
            )

            self._logic.reset()
            self._last_processed_step = None

            # Per-episode safety observations must not be carried
            # into a newly started execution. Require fresh runtime
            # measurements before evaluating the new episode.
            self._maximum_principal_sigma = None
            self._predicted_clearance = None
            self._tracking_error = None
            self._latest_perception_step = None
            self._perception_waiting_for_step = False

        self._execution_steps = step

        if self._perception_waiting_for_step:
            self._latest_perception_step = step
            self._perception_waiting_for_step = False

        if self._last_processed_step == step:
            return

        self._process_current_snapshot()

    def _inputs_ready(self) -> bool:
        """Return whether a complete runtime snapshot is available."""
        return all(
            value is not None
            for value in (
                self._joint_names,
                self._joint_positions,
                self._maximum_principal_sigma,
                self._predicted_clearance,
                self._tracking_error,
                self._execution_steps,
                self._latest_perception_step,
            )
        )

    def _process_current_snapshot(self) -> None:
        """Evaluate the latest complete ROS runtime state."""
        if not self._inputs_ready():
            return

        step = int(
            self._execution_steps
        )

        try:
            snapshot = build_runtime_snapshot(
                names=self._joint_names,
                positions=self._joint_positions,
                step_index=step,
                latest_perception_step=int(
                    self._latest_perception_step
                ),
                maximum_principal_sigma=float(
                    self._maximum_principal_sigma
                ),
                predicted_clearance=float(
                    self._predicted_clearance
                ),
                tracking_error=float(
                    self._tracking_error
                ),
                execution_steps=step,
            )

            decision = self._logic.process(
                snapshot
            )

        except (
            RuntimeError,
            ValueError,
        ) as error:
            self.get_logger().error(
                'Supervisor snapshot processing failed: '
                f'{error}'
            )

            return

        self._last_processed_step = step

        if decision is None:
            return

        self._publish_status(
            decision
        )

        command = command_from_decision(
            decision
        )

        if command is not None:
            message = String()
            message.data = command

            self._command_publisher.publish(
                message
            )

            self.get_logger().warning(
                'Supervisor command: '
                f'{command.upper()}'
            )

    def _publish_status(
        self,
        decision,
    ) -> None:
        """Publish the auditable Phase 7 controller state."""
        message = SafetyStatus()

        message.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        message.header.frame_id = 'robot_base'

        message.state = decision.state.value

        event = decision.triggering_event

        if (
            event is None
            and decision.active_events
        ):
            event = highest_severity_event(
                decision.active_events
            )

        if event is None:
            message.highest_severity = 'none'
            message.active_hazard = 'none'
            message.message = (
                'No active runtime safety hazards.'
            )
        else:
            message.highest_severity = (
                event.severity.value
            )

            message.active_hazard = (
                event.hazard.value
            )

            message.message = (
                event.message
            )

        controller = self._logic.controller

        message.replan_count = int(
            controller.replan_count
        )

        message.reacquisition_count = int(
            controller.reacquisition_count
        )

        message.recovery_count = int(
            controller.recovery_count
        )

        message.stop_count = int(
            controller.stop_count
        )

        message.failure_count = int(
            controller.failure_count
        )

        self._status_publisher.publish(
            message
        )


def main(
    args=None,
) -> None:
    """Run the autonomous ROS supervisor."""
    rclpy.init(
        args=args
    )

    node = AutonomousSupervisorNode()

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
