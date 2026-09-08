"""Execute supervised research-core navigation trajectories through ROS 2."""

from __future__ import annotations

import importlib
import os
import threading
import time

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import UInt32
from surgical_navigation_interfaces.action import ExecuteNavigation
from surgical_navigation_interfaces.msg import EstimatedStructure
from surgical_navigation_interfaces.srv import RequestReplan
from surgical_navigation_ros.core_adapter import (
    core_path_to_ros_positions,
    ROS_JOINT_NAMES,
    ros_joint_state_to_core_q,
    RuntimeSafetyConfig,
)
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)
from visualization_msgs.msg import Marker


# Importing core_adapter above establishes the repository root on sys.path.
_workspace_module = importlib.import_module(
    'src.geometry.workspace'
)

_instrument_module = importlib.import_module(
    'src.robotics.instrument'
)

_planner_module = importlib.import_module(
    'src.robotics.planner'
)

_trajectory_module = importlib.import_module(
    'src.robotics.trajectory_execution'
)

_safety_module = importlib.import_module(
    'src.robotics.safety'
)


SphericalStructure = (
    _workspace_module.SphericalStructure
)

SurgicalInstrument = (
    _instrument_module.SurgicalInstrument
)

plan_rrt = (
    _planner_module.plan_rrt
)

shortcut_path = (
    _planner_module.shortcut_path
)

time_parameterise_path = (
    _trajectory_module.time_parameterise_path
)

evaluate_instrument_safety = (
    _safety_module.evaluate_instrument_safety
)


INSTRUMENT_RADIUS = 0.006
PROXIMAL_LENGTH = 0.10

PLANNER_MAX_ITERATIONS = 5000
PLANNER_STEP_SIZE = 0.08
PLANNER_GOAL_BIAS = 0.15
PLANNER_EDGE_RESOLUTION = 20
PLANNER_SEED = 7

SHORTCUT_EDGE_RESOLUTION = 30
SHORTCUT_ATTEMPTS = 300
SHORTCUT_SEED = 11

ACTION_NAME = '/navigation/execute_navigation'

TRAJECTORY_ACTION_NAME = (
    '/joint_trajectory_controller/follow_joint_trajectory'
)

SUPERVISOR_COMMAND_TOPIC = (
    '/navigation/supervisor_command'
)

REPLAN_SERVICE_NAME = (
    '/navigation/request_replan'
)

EXECUTION_STEPS_TOPIC = (
    '/navigation/execution_steps'
)

PLANNED_PATH_TOPIC = (
    '/navigation/planned_path_marker'
)

ESTIMATED_STRUCTURE_TOPIC = (
    '/navigation/estimated_structure'
)

TRACKING_ERROR_TOPIC = (
    '/navigation/tracking_error'
)

PREDICTED_CLEARANCE_TOPIC = (
    '/navigation/predicted_clearance'
)

CONTROLLER_STATE_TOPIC = (
    '/joint_trajectory_controller/controller_state'
)

EXECUTION_LOOP_PERIOD_SECONDS = 0.05
EXECUTION_STEP_PERIOD_SECONDS = 0.20

INTERVENTION_TIMEOUT_SECONDS = 8.0
CONTROLLER_CANCEL_TIMEOUT_SECONDS = 2.0

MAX_REPLAN_ATTEMPTS = 8

VALID_SUPERVISOR_COMMANDS = (
    'replan',
    'reacquire',
    'recover',
    'stop',
)


def make_phase2_instrument() -> SurgicalInstrument:
    """Create the frozen Phase 2 RCM instrument."""
    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_phase2_structures() -> tuple[
    SphericalStructure,
    ...,
]:
    """Create the established Phase 2 protected structures."""
    return (
        SphericalStructure(
            centre=np.asarray(
                [
                    0.14,
                    0.04,
                    0.00,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.asarray(
                [
                    0.18,
                    -0.06,
                    0.02,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def target_position_from_pose(
    pose: PoseStamped,
) -> np.ndarray:
    """Extract a finite Cartesian target from a PoseStamped."""
    target = np.asarray(
        [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        ],
        dtype=float,
    )

    if target.shape != (3,):
        raise ValueError(
            'Target position must have shape (3,).'
        )

    if not np.all(
        np.isfinite(
            target
        )
    ):
        raise ValueError(
            'Target position must contain finite values.'
        )

    return target


def core_quaternion(
    q: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Return the ROS quaternion for a Phase 2 configuration."""
    configuration = np.asarray(
        q,
        dtype=float,
    )

    if configuration.shape != (4,):
        raise ValueError(
            'Configuration must have shape (4,).'
        )

    if not np.all(
        np.isfinite(
            configuration
        )
    ):
        raise ValueError(
            'Configuration must contain finite values.'
        )

    yaw = float(
        configuration[0]
    )

    pitch = float(
        configuration[1]
    )

    roll = float(
        configuration[3]
    )

    ros_pitch = -pitch

    half_roll = 0.5 * roll
    half_pitch = 0.5 * ros_pitch
    half_yaw = 0.5 * yaw

    cr = np.cos(
        half_roll
    )
    sr = np.sin(
        half_roll
    )

    cp = np.cos(
        half_pitch
    )
    sp = np.sin(
        half_pitch
    )

    cy = np.cos(
        half_yaw
    )
    sy = np.sin(
        half_yaw
    )

    qx = (
        sr * cp * cy
        - cr * sp * sy
    )

    qy = (
        cr * sp * cy
        + sr * cp * sy
    )

    qz = (
        cr * cp * sy
        - sr * sp * cy
    )

    qw = (
        cr * cp * cy
        + sr * sp * sy
    )

    return (
        float(qx),
        float(qy),
        float(qz),
        float(qw),
    )


def make_follow_joint_trajectory(
    timed_trajectory,
) -> JointTrajectory:
    """Convert a core timed trajectory to a ROS position trajectory."""
    times = np.asarray(
        timed_trajectory.times,
        dtype=float,
    )

    positions = core_path_to_ros_positions(
        timed_trajectory.positions
    )

    if times.ndim != 1:
        raise ValueError(
            'Trajectory times must be one-dimensional.'
        )

    if (
        positions.ndim != 2
        or positions.shape[1] != 4
    ):
        raise ValueError(
            'Trajectory positions must have shape (N, 4).'
        )

    if len(times) != len(positions):
        raise ValueError(
            'Trajectory times and positions must have '
            'matching lengths.'
        )

    if len(times) < 2:
        raise ValueError(
            'Timed trajectory must contain at least two samples.'
        )

    if not np.all(
        np.isfinite(
            times
        )
    ):
        raise ValueError(
            'Trajectory times must contain finite values.'
        )

    if not np.all(
        np.isfinite(
            positions
        )
    ):
        raise ValueError(
            'Trajectory positions must contain finite values.'
        )

    if np.any(
        np.diff(
            times
        ) <= 0.0
    ):
        raise ValueError(
            'Trajectory times must be strictly increasing.'
        )

    trajectory = JointTrajectory()

    trajectory.joint_names = list(
        ROS_JOINT_NAMES
    )

    for index in range(
        1,
        len(times),
    ):
        point = JointTrajectoryPoint()

        point.positions = [
            float(value)
            for value in positions[
                index
            ]
        ]

        point.time_from_start = Duration(
            seconds=float(
                times[index]
            )
        ).to_msg()

        trajectory.points.append(
            point
        )

    if not trajectory.points:
        raise ValueError(
            'ROS trajectory contains no executable points.'
        )

    return trajectory


class NavigationExecutionNode(Node):
    """Execute Phase 2 trajectories under Phase 7 ROS supervision."""

    def __init__(
        self,
    ) -> None:
        """Create the supervised navigation execution node."""
        super().__init__(
            'navigation_execution'
        )

        self._callback_group = (
            ReentrantCallbackGroup()
        )

        self._instrument = (
            make_phase2_instrument()
        )

        self._structures = (
            make_phase2_structures()
        )

        self._safety_config = (
            RuntimeSafetyConfig()
        )

        self._lock = threading.Lock()

        self._latest_joint_positions: np.ndarray | None = None

        self._latest_sigma: float | None = None
        self._latest_tracking_error: float | None = None

        self._perception_generation = 0
        self._tracking_generation = 0

        self._execution_active = False

        self._pending_command: str | None = None
        self._pending_replan_target: np.ndarray | None = None
        self._active_intervention: str | None = None

        self._execution_step = 0
        self._last_execution_step_time = (
            time.monotonic()
        )

        self._joint_subscription = (
            self.create_subscription(
                JointState,
                '/joint_states',
                self._joint_state_callback,
                10,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self._supervisor_subscription = (
            self.create_subscription(
                String,
                SUPERVISOR_COMMAND_TOPIC,
                self._supervisor_command_callback,
                10,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self._estimated_structure_subscription = (
            self.create_subscription(
                EstimatedStructure,
                ESTIMATED_STRUCTURE_TOPIC,
                self._estimated_structure_callback,
                10,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self._tracking_error_subscription = (
            self.create_subscription(
                Float64,
                TRACKING_ERROR_TOPIC,
                self._tracking_error_callback,
                10,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self._controller_state_subscription = (
            self.create_subscription(
                JointTrajectoryControllerState,
                CONTROLLER_STATE_TOPIC,
                self._controller_state_callback,
                10,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self._predicted_clearance_publisher = (
            self.create_publisher(
                Float64,
                PREDICTED_CLEARANCE_TOPIC,
                10,
            )
        )

        self._tracking_error_publisher = (
            self.create_publisher(
                Float64,
                TRACKING_ERROR_TOPIC,
                10,
            )
        )

        self._execution_step_publisher = (
            self.create_publisher(
                UInt32,
                EXECUTION_STEPS_TOPIC,
                10,
            )
        )

        self._planned_path_publisher = (
            self.create_publisher(
                Marker,
                PLANNED_PATH_TOPIC,
                10,
            )
        )

        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            TRAJECTORY_ACTION_NAME,
            callback_group=(
                self._callback_group
            ),
        )

        self._action_server = ActionServer(
            self,
            ExecuteNavigation,
            ACTION_NAME,
            execute_callback=(
                self._execute_callback
            ),
            goal_callback=(
                self._goal_callback
            ),
            cancel_callback=(
                self._cancel_callback
            ),
            callback_group=(
                self._callback_group
            ),
        )

        self._replan_service = (
            self.create_service(
                RequestReplan,
                REPLAN_SERVICE_NAME,
                self._request_replan_callback,
                callback_group=(
                    self._callback_group
                ),
            )
        )

        self.get_logger().info(
            'Supervised navigation execution node ready.'
        )

        self.get_logger().info(
            'Phase 7 commands enabled: '
            'REPLAN, REACQUIRE, RECOVER, STOP.'
        )

        self.get_logger().info(
            'Target orientation is not used by Phase 2 '
            'position-only IK; goal roll is fixed to zero.'
        )

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        """Store the latest valid measured configuration."""
        try:
            core_q = (
                ros_joint_state_to_core_q(
                    message.name,
                    message.position,
                )
            )
        except ValueError:
            return

        with self._lock:
            self._latest_joint_positions = (
                core_q.copy()
            )

    def _estimated_structure_callback(
        self,
        message: EstimatedStructure,
    ) -> None:
        """Store the latest perception uncertainty."""
        sigma = float(
            message.maximum_principal_sigma
        )

        if not np.isfinite(
            sigma
        ):
            return

        with self._lock:
            self._latest_sigma = sigma
            self._perception_generation += 1

    def _tracking_error_callback(
        self,
        message: Float64,
    ) -> None:
        """Store the latest trajectory-tracking error."""
        value = float(
            message.data
        )

        if not np.isfinite(
            value
        ):
            return

        with self._lock:
            self._latest_tracking_error = value
            self._tracking_generation += 1

    def _controller_state_callback(
        self,
        message: JointTrajectoryControllerState,
    ) -> None:
        """Publish task-space controller tracking error in metres."""
        try:
            reference_q = ros_joint_state_to_core_q(
                message.joint_names,
                message.reference.positions,
            )

            feedback_q = ros_joint_state_to_core_q(
                message.joint_names,
                message.feedback.positions,
            )

            reference_position = (
                self._instrument.forward_position(
                    reference_q
                )
            )

            feedback_position = (
                self._instrument.forward_position(
                    feedback_q
                )
            )

        except ValueError:
            return

        tracking_error = float(
            np.linalg.norm(
                reference_position
                - feedback_position
            )
        )

        if not np.isfinite(tracking_error):
            return

        message_out = Float64()
        message_out.data = tracking_error

        self._tracking_error_publisher.publish(
            message_out
        )

    def _current_core_q(
        self,
    ) -> np.ndarray:
        """Return the latest measured configuration in core order."""
        with self._lock:
            if self._latest_joint_positions is None:
                raise RuntimeError(
                    'No valid joint state has been received.'
                )

            return (
                self._latest_joint_positions
                .copy()
            )

    def _pose_from_core_q(
        self,
        q: np.ndarray,
    ) -> PoseStamped:
        """Create robot_base pose feedback from a core configuration."""
        position = (
            self._instrument
            .forward_position(
                q
            )
        )

        quaternion = core_quaternion(
            q
        )

        pose = PoseStamped()

        pose.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        pose.header.frame_id = (
            'robot_base'
        )

        pose.pose.position.x = float(
            position[0]
        )

        pose.pose.position.y = float(
            position[1]
        )

        pose.pose.position.z = float(
            position[2]
        )

        (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ) = quaternion

        return pose

    def _publish_feedback(
        self,
        goal_handle,
        *,
        progress: float,
        state: str,
    ) -> None:
        """Publish action progress and measured instrument pose."""
        feedback = (
            ExecuteNavigation.Feedback()
        )

        feedback.progress = float(
            np.clip(
                progress,
                0.0,
                1.0,
            )
        )

        feedback.safety_state = str(
            state
        )

        try:
            current_q = (
                self._current_core_q()
            )

            feedback.current_pose = (
                self._pose_from_core_q(
                    current_q
                )
            )

        except (
            RuntimeError,
            ValueError,
        ):
            feedback.current_pose = (
                PoseStamped()
            )

            feedback.current_pose.header.frame_id = (
                'robot_base'
            )

        goal_handle.publish_feedback(
            feedback
        )

    def _publish_execution_step(
        self,
        *,
        force: bool = False,
    ) -> None:
        """Publish a logical runtime-supervision execution step."""
        now = time.monotonic()

        if (
            not force
            and (
                now
                - self._last_execution_step_time
            )
            < EXECUTION_STEP_PERIOD_SECONDS
        ):
            return

        if force:
            self._execution_step = 0
        else:
            self._execution_step += 1

        message = UInt32()

        message.data = int(
            self._execution_step
        )

        self._execution_step_publisher.publish(
            message
        )

        self._last_execution_step_time = now

    def _queue_command(
        self,
        command: str,
        *,
        target: np.ndarray | None = None,
    ) -> bool:
        """Queue one intervention command for the active execution."""
        normalized = str(
            command
        ).strip().lower()

        if normalized not in VALID_SUPERVISOR_COMMANDS:
            return False

        with self._lock:
            if not self._execution_active:
                return False

            if normalized == 'stop':
                self._pending_command = 'stop'
                self._pending_replan_target = None
                return True

            if self._active_intervention is not None:
                return False

            if self._pending_command is not None:
                return False

            self._pending_command = normalized

            if target is None:
                self._pending_replan_target = None
            else:
                self._pending_replan_target = (
                    np.asarray(
                        target,
                        dtype=float,
                    ).copy()
                )

            return True

    def _supervisor_command_callback(
        self,
        message: String,
    ) -> None:
        """Receive an intervention command from the Phase 7 supervisor."""
        command = str(
            message.data
        ).strip().lower()

        accepted = self._queue_command(
            command
        )

        if accepted:
            self.get_logger().warning(
                'Received supervisor command: '
                f'{command.upper()}'
            )

    def _take_pending_command(
        self,
    ) -> tuple[
        str | None,
        np.ndarray | None,
    ]:
        """Consume one pending intervention command."""
        with self._lock:
            command = self._pending_command

            target = self._pending_replan_target

            self._pending_command = None
            self._pending_replan_target = None

            if command is not None:
                self._active_intervention = command

            if target is not None:
                target = target.copy()

            return (
                command,
                target,
            )

    def _take_pending_stop(
        self,
    ) -> bool:
        """Consume a STOP request while another intervention is active."""
        with self._lock:
            if self._pending_command != 'stop':
                return False

            self._pending_command = None
            self._pending_replan_target = None
            self._active_intervention = 'stop'

            return True

    def _finish_intervention(
        self,
    ) -> None:
        """Mark the current intervention as resolved."""
        with self._lock:
            self._active_intervention = None

    def _clear_command_state(
        self,
    ) -> None:
        """Clear intervention state at execution boundaries."""
        with self._lock:
            self._pending_command = None
            self._pending_replan_target = None
            self._active_intervention = None

    def _request_replan_callback(
        self,
        request,
        response,
    ):
        """Queue an explicit RequestReplan service request."""
        with self._lock:
            execution_active = (
                self._execution_active
            )

        if not execution_active:
            response.accepted = False
            response.message = (
                'No ExecuteNavigation goal is currently active.'
            )

            return response

        frame = (
            request
            .target
            .header
            .frame_id
        )

        if frame not in (
            '',
            'robot_base',
        ):
            response.accepted = False
            response.message = (
                'Replan target must be expressed in robot_base.'
            )

            return response

        try:
            target = target_position_from_pose(
                request.target
            )

            self._instrument.inverse_position(
                target,
                roll=0.0,
            )

        except ValueError as error:
            response.accepted = False
            response.message = str(
                error
            )

            return response

        queued = self._queue_command(
            'replan',
            target=target,
        )

        response.accepted = bool(
            queued
        )

        if queued:
            response.message = (
                'Replan request queued.'
            )

            self.get_logger().warning(
                'RequestReplan service queued a new target.'
            )
        else:
            response.message = (
                'Unable to queue replan while another '
                'intervention is pending.'
            )

        return response

    def _goal_callback(
        self,
        goal_request,
    ) -> GoalResponse:
        """Validate an incoming navigation goal."""
        with self._lock:
            execution_active = (
                self._execution_active
            )

            have_joint_state = (
                self._latest_joint_positions
                is not None
            )

        if execution_active:
            self.get_logger().warning(
                'Rejecting goal because another navigation '
                'execution is active.'
            )

            return GoalResponse.REJECT

        if not have_joint_state:
            self.get_logger().warning(
                'Rejecting goal because no valid joint '
                'state has been received.'
            )

            return GoalResponse.REJECT

        frame = (
            goal_request
            .target
            .header
            .frame_id
        )

        if frame not in (
            '',
            'robot_base',
        ):
            self.get_logger().warning(
                'Rejecting target outside robot_base frame.'
            )

            return GoalResponse.REJECT

        try:
            target = target_position_from_pose(
                goal_request.target
            )

            self._instrument.inverse_position(
                target,
                roll=0.0,
            )

        except ValueError as error:
            self.get_logger().warning(
                'Rejecting invalid or unreachable target: '
                f'{error}'
            )

            return GoalResponse.REJECT

        if not (
            self._trajectory_client
            .server_is_ready()
        ):
            self.get_logger().warning(
                'Rejecting goal because the trajectory '
                'controller action server is not ready.'
            )

            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_callback(
        self,
        _goal_handle,
    ) -> CancelResponse:
        """Accept cancellation of active navigation."""
        return CancelResponse.ACCEPT

    def _minimum_path_safety_clearance(
        self,
        path,
    ) -> float:
        """Return minimum predicted safety clearance along a path."""
        minimum_clearance = float('inf')

        for configuration in path:
            q = np.asarray(
                configuration,
                dtype=float,
            )

            shaft_start, shaft_end = (
                self._instrument.shaft_segment(
                    q,
                    proximal_length=(
                        PROXIMAL_LENGTH
                    ),
                )
            )

            evaluation = (
                evaluate_instrument_safety(
                    shaft_start=shaft_start,
                    shaft_end=shaft_end,
                    structures=self._structures,
                    instrument_radius=(
                        INSTRUMENT_RADIUS
                    ),
                )
            )

            minimum_clearance = min(
                minimum_clearance,
                float(
                    evaluation
                    .minimum_safety_clearance
                ),
            )

        return float(
            minimum_clearance
        )

    def _publish_predicted_clearance(
        self,
        path,
    ) -> float:
        """Publish planner-predicted minimum safety clearance."""
        clearance = (
            self._minimum_path_safety_clearance(
                path
            )
        )

        message = Float64()
        message.data = clearance

        self._predicted_clearance_publisher.publish(
            message
        )

        self.get_logger().info(
            'Predicted minimum safety clearance: '
            f'{clearance:.6f} m.'
        )

        return clearance

    def _publish_planned_path(
        self,
        path,
    ) -> None:
        """Publish the real smoothed planner path as tool-tip positions."""
        marker = Marker()

        marker.header.frame_id = 'robot_base'
        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = 'planned_navigation_path'
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.0025

        marker.color.r = 0.05
        marker.color.g = 0.80
        marker.color.b = 1.00
        marker.color.a = 1.00

        for configuration in path:
            q = np.asarray(
                configuration,
                dtype=float,
            )

            tip_position = (
                self._instrument.forward_position(
                    q
                )
            )

            point = Point()

            point.x = float(
                tip_position[0]
            )
            point.y = float(
                tip_position[1]
            )
            point.z = float(
                tip_position[2]
            )

            marker.points.append(
                point
            )

        self._planned_path_publisher.publish(
            marker
        )

        self.get_logger().info(
            'Published planned tool-tip path with '
            f'{len(marker.points)} waypoints.'
        )

    def _plan(
        self,
        *,
        start_q: np.ndarray,
        target_position: np.ndarray,
    ):
        """Plan and time-parameterise one Phase 2 trajectory."""
        goal_q = (
            self._instrument
            .inverse_position(
                target_position,
                roll=0.0,
            )
        )

        planning_result = plan_rrt(
            instrument=self._instrument,
            start_q=start_q,
            goal_q=goal_q,
            structures=self._structures,
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            max_iterations=(
                PLANNER_MAX_ITERATIONS
            ),
            step_size=(
                PLANNER_STEP_SIZE
            ),
            goal_bias=(
                PLANNER_GOAL_BIAS
            ),
            edge_resolution=(
                PLANNER_EDGE_RESOLUTION
            ),
            seed=(
                PLANNER_SEED
            ),
        )

        if not planning_result.success:
            raise RuntimeError(
                'Phase 2 RRT failed to produce a safe path.'
            )

        smoothed_path = shortcut_path(
            instrument=self._instrument,
            path=planning_result.path,
            structures=self._structures,
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            edge_resolution=(
                SHORTCUT_EDGE_RESOLUTION
            ),
            attempts=(
                SHORTCUT_ATTEMPTS
            ),
            seed=(
                SHORTCUT_SEED
            ),
        )

        speed_scale_text = os.environ.get(
            'SURGICAL_NAVIGATION_SPEED_SCALE',
            '1.0',
        )

        try:
            speed_scale = float(
                speed_scale_text
            )
        except ValueError as exc:
            raise ValueError(
                'SURGICAL_NAVIGATION_SPEED_SCALE '
                'must be numeric.'
            ) from exc

        if (
            not np.isfinite(
                speed_scale
            )
            or speed_scale <= 0.0
            or speed_scale > 1.0
        ):
            raise ValueError(
                'SURGICAL_NAVIGATION_SPEED_SCALE '
                'must be in the interval (0, 1].'
            )

        base_time_config = (
            _trajectory_module
            .TimeParameterisationConfig()
        )

        scaled_velocities = tuple(
            float(value)
            * speed_scale
            for value in (
                base_time_config
                .maximum_joint_velocities
            )
        )

        time_config = (
            _trajectory_module
            .TimeParameterisationConfig(
                maximum_joint_velocities=(
                    scaled_velocities
                ),
                sample_period=(
                    base_time_config
                    .sample_period
                ),
                minimum_segment_duration=(
                    base_time_config
                    .minimum_segment_duration
                ),
            )
        )

        timed_trajectory = (
            time_parameterise_path(
                self._instrument,
                smoothed_path,
                config=time_config,
            )
        )

        self._publish_predicted_clearance(
            smoothed_path
        )

        self._publish_planned_path(
            smoothed_path
        )

        return (
            planning_result,
            smoothed_path,
            timed_trajectory,
        )

    async def _cancel_controller_goal(
        self,
        controller_goal_handle,
        result_future,
    ) -> None:
        """Cancel the active ros2_control trajectory."""
        try:
            cancel_future = (
                controller_goal_handle
                .cancel_goal_async()
            )

            await cancel_future

        except Exception as error:
            self.get_logger().warning(
                'Controller cancellation request failed: '
                f'{error}'
            )

            return

        deadline = (
            time.monotonic()
            + CONTROLLER_CANCEL_TIMEOUT_SECONDS
        )

        while (
            not result_future.done()
            and time.monotonic() < deadline
        ):
            time.sleep(
                EXECUTION_LOOP_PERIOD_SECONDS
            )

    def _wait_for_reacquisition(
        self,
        goal_handle,
    ) -> str:
        """Wait for a fresh observation below the recovery threshold."""
        with self._lock:
            initial_generation = (
                self._perception_generation
            )

        deadline = (
            time.monotonic()
            + INTERVENTION_TIMEOUT_SECONDS
        )

        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return 'cancelled'

            if self._take_pending_stop():
                return 'stop'

            self._publish_execution_step()

            with self._lock:
                generation = (
                    self._perception_generation
                )

                sigma = (
                    self._latest_sigma
                )

            if (
                generation > initial_generation
                and sigma is not None
                and sigma
                < self._safety_config
                .uncertainty_reacquire_sigma
            ):
                self.get_logger().info(
                    'Perception reacquisition succeeded: '
                    f'sigma={sigma:.6f}.'
                )

                return 'ready'

            time.sleep(
                EXECUTION_LOOP_PERIOD_SECONDS
            )

        return 'timeout'

    def _wait_for_recovery(
        self,
        goal_handle,
    ) -> str:
        """Wait for fresh tracking data below the recovery threshold."""
        with self._lock:
            initial_generation = (
                self._tracking_generation
            )

        deadline = (
            time.monotonic()
            + INTERVENTION_TIMEOUT_SECONDS
        )

        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return 'cancelled'

            if self._take_pending_stop():
                return 'stop'

            self._publish_execution_step()

            with self._lock:
                generation = (
                    self._tracking_generation
                )

                tracking_error = (
                    self._latest_tracking_error
                )

            if (
                generation > initial_generation
                and tracking_error is not None
                and tracking_error
                < self._safety_config
                .tracking_error_recovery_threshold
            ):
                self.get_logger().info(
                    'Tracking recovery succeeded: '
                    f'error={tracking_error:.6f}.'
                )

                return 'ready'

            time.sleep(
                EXECUTION_LOOP_PERIOD_SECONDS
            )

        return 'timeout'

    def _resolve_intervention(
        self,
        *,
        command: str,
        requested_target: np.ndarray | None,
        current_target: np.ndarray,
        goal_handle,
    ) -> tuple[
        str,
        np.ndarray,
    ]:
        """Resolve one supervisor intervention."""
        if command == 'stop':
            self.get_logger().error(
                'STOP intervention received; '
                'navigation execution halted.'
            )

            return (
                'stop',
                current_target,
            )

        if command == 'replan':
            self._publish_feedback(
                goal_handle,
                progress=0.30,
                state='replanning',
            )

            if requested_target is not None:
                current_target = (
                    requested_target.copy()
                )

            self.get_logger().warning(
                'REPLAN intervention: planning again '
                'from the live measured joint state.'
            )

            self._finish_intervention()

            return (
                'replan',
                current_target,
            )

        if command == 'reacquire':
            self._publish_feedback(
                goal_handle,
                progress=0.30,
                state='reacquiring',
            )

            self.get_logger().warning(
                'REACQUIRE intervention: execution paused '
                'until fresh lower-uncertainty perception arrives.'
            )

            outcome = (
                self._wait_for_reacquisition(
                    goal_handle
                )
            )

            self._finish_intervention()

            if outcome == 'ready':
                return (
                    'replan',
                    current_target,
                )

            return (
                outcome,
                current_target,
            )

        if command == 'recover':
            self._publish_feedback(
                goal_handle,
                progress=0.30,
                state='recovering',
            )

            self.get_logger().warning(
                'RECOVER intervention: execution paused '
                'until fresh tracking error returns below threshold.'
            )

            outcome = self._wait_for_recovery(
                goal_handle
            )

            self._finish_intervention()

            if outcome == 'ready':
                return (
                    'replan',
                    current_target,
                )

            return (
                outcome,
                current_target,
            )

        self._finish_intervention()

        return (
            'failed',
            current_target,
        )

    async def _execute_controller_trajectory(
        self,
        *,
        goal_handle,
        trajectory_message: JointTrajectory,
        expected_duration: float,
    ) -> tuple[
        str,
        np.ndarray | None,
        object | None,
    ]:
        """Execute one trajectory until completion or intervention."""
        controller_goal = (
            FollowJointTrajectory.Goal()
        )

        controller_goal.trajectory = (
            trajectory_message
        )

        send_future = (
            self._trajectory_client
            .send_goal_async(
                controller_goal
            )
        )

        controller_goal_handle = (
            await send_future
        )

        if not controller_goal_handle.accepted:
            return (
                'controller_rejected',
                None,
                None,
            )

        self.get_logger().info(
            'JointTrajectoryController accepted trajectory.'
        )

        result_future = (
            controller_goal_handle
            .get_result_async()
        )

        start_time = time.monotonic()

        while not result_future.done():
            if goal_handle.is_cancel_requested:
                await self._cancel_controller_goal(
                    controller_goal_handle,
                    result_future,
                )

                return (
                    'cancelled',
                    None,
                    None,
                )

            command, requested_target = (
                self._take_pending_command()
            )

            if command is not None:
                await self._cancel_controller_goal(
                    controller_goal_handle,
                    result_future,
                )

                self.get_logger().warning(
                    'Active trajectory cancelled for '
                    f'{command.upper()} intervention.'
                )

                return (
                    command,
                    requested_target,
                    None,
                )

            self._publish_execution_step()

            elapsed = (
                time.monotonic()
                - start_time
            )

            if expected_duration > 0.0:
                execution_fraction = (
                    elapsed
                    / expected_duration
                )
            else:
                execution_fraction = 1.0

            progress = (
                0.20
                + 0.79
                * float(
                    np.clip(
                        execution_fraction,
                        0.0,
                        1.0,
                    )
                )
            )

            self._publish_feedback(
                goal_handle,
                progress=progress,
                state='executing',
            )

            time.sleep(
                EXECUTION_LOOP_PERIOD_SECONDS
            )

        wrapped_result = (
            result_future.result()
        )

        return (
            'completed',
            None,
            wrapped_result.result,
        )

    @staticmethod
    def _result(
        *,
        success: bool,
        terminal_state: str,
        message: str,
    ):
        """Create an ExecuteNavigation result."""
        result = (
            ExecuteNavigation.Result()
        )

        result.success = bool(
            success
        )

        result.terminal_state = str(
            terminal_state
        )

        result.message = str(
            message
        )

        return result

    async def _execute_callback(
        self,
        goal_handle,
    ):
        """Plan and execute one supervised navigation goal."""
        with self._lock:
            self._execution_active = True

        self._clear_command_state()

        self._execution_step = 0

        self._last_execution_step_time = (
            time.monotonic()
        )

        self._publish_execution_step(
            force=True
        )

        current_target = (
            target_position_from_pose(
                goal_handle
                .request
                .target
            )
        )

        replan_attempts = 0

        try:
            self.get_logger().info(
                'ExecuteNavigation goal accepted.'
            )

            while True:
                if (
                    replan_attempts
                    > MAX_REPLAN_ATTEMPTS
                ):
                    goal_handle.abort()

                    return self._result(
                        success=False,
                        terminal_state='failed',
                        message=(
                            'Maximum supervised replanning '
                            'attempts exceeded.'
                        ),
                    )

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()

                    return self._result(
                        success=False,
                        terminal_state='cancelled',
                        message=(
                            'Navigation execution was cancelled.'
                        ),
                    )

                command, requested_target = (
                    self._take_pending_command()
                )

                if command is not None:
                    (
                        intervention_outcome,
                        current_target,
                    ) = self._resolve_intervention(
                        command=command,
                        requested_target=(
                            requested_target
                        ),
                        current_target=(
                            current_target
                        ),
                        goal_handle=(
                            goal_handle
                        ),
                    )

                    if intervention_outcome == 'stop':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='stopped',
                            message=(
                                'Navigation stopped by the '
                                'autonomous safety supervisor.'
                            ),
                        )

                    if intervention_outcome == 'cancelled':
                        goal_handle.canceled()

                        return self._result(
                            success=False,
                            terminal_state='cancelled',
                            message=(
                                'Navigation execution was cancelled.'
                            ),
                        )

                    if intervention_outcome == 'timeout':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='failed',
                            message=(
                                'Safety intervention timed out '
                                'without recovery.'
                            ),
                        )

                    if intervention_outcome == 'replan':
                        replan_attempts += 1
                        continue

                self._publish_feedback(
                    goal_handle,
                    progress=0.05,
                    state=(
                        'planning'
                        if replan_attempts == 0
                        else 'replanning'
                    ),
                )

                start_q = (
                    self._current_core_q()
                )

                try:
                    (
                        planning_result,
                        smoothed_path,
                        timed_trajectory,
                    ) = self._plan(
                        start_q=start_q,
                        target_position=(
                            current_target
                        ),
                    )

                except (
                    RuntimeError,
                    ValueError,
                ) as error:
                    goal_handle.abort()

                    self.get_logger().error(
                        'Navigation planning failed: '
                        f'{error}'
                    )

                    return self._result(
                        success=False,
                        terminal_state='failed',
                        message=str(
                            error
                        ),
                    )

                self.get_logger().info(
                    'Phase 2 planner succeeded: '
                    f'{planning_result.iterations} iterations, '
                    f'{len(smoothed_path)} smoothed waypoints.'
                )

                trajectory_message = (
                    make_follow_joint_trajectory(
                        timed_trajectory
                    )
                )

                self.get_logger().info(
                    'Generated ROS trajectory with '
                    f'{len(trajectory_message.points)} points.'
                )

                command, requested_target = (
                    self._take_pending_command()
                )

                if command is not None:
                    (
                        intervention_outcome,
                        current_target,
                    ) = self._resolve_intervention(
                        command=command,
                        requested_target=(
                            requested_target
                        ),
                        current_target=(
                            current_target
                        ),
                        goal_handle=(
                            goal_handle
                        ),
                    )

                    if intervention_outcome == 'stop':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='stopped',
                            message=(
                                'Navigation stopped before '
                                'trajectory execution.'
                            ),
                        )

                    if intervention_outcome == 'replan':
                        replan_attempts += 1
                        continue

                    if intervention_outcome == 'cancelled':
                        goal_handle.canceled()

                        return self._result(
                            success=False,
                            terminal_state='cancelled',
                            message=(
                                'Navigation execution was cancelled.'
                            ),
                        )

                    if intervention_outcome == 'timeout':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='failed',
                            message=(
                                'Safety intervention timed out.'
                            ),
                        )

                self._publish_feedback(
                    goal_handle,
                    progress=0.20,
                    state='executing',
                )

                (
                    execution_outcome,
                    requested_target,
                    controller_result,
                ) = await self._execute_controller_trajectory(
                    goal_handle=goal_handle,
                    trajectory_message=(
                        trajectory_message
                    ),
                    expected_duration=float(
                        timed_trajectory.times[-1]
                    ),
                )

                if execution_outcome in (
                    'replan',
                    'reacquire',
                    'recover',
                    'stop',
                ):
                    (
                        intervention_outcome,
                        current_target,
                    ) = self._resolve_intervention(
                        command=(
                            execution_outcome
                        ),
                        requested_target=(
                            requested_target
                        ),
                        current_target=(
                            current_target
                        ),
                        goal_handle=(
                            goal_handle
                        ),
                    )

                    if intervention_outcome == 'replan':
                        replan_attempts += 1
                        continue

                    if intervention_outcome == 'stop':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='stopped',
                            message=(
                                'Navigation stopped by the '
                                'autonomous safety supervisor.'
                            ),
                        )

                    if intervention_outcome == 'cancelled':
                        goal_handle.canceled()

                        return self._result(
                            success=False,
                            terminal_state='cancelled',
                            message=(
                                'Navigation execution was cancelled.'
                            ),
                        )

                    if intervention_outcome == 'timeout':
                        goal_handle.abort()

                        return self._result(
                            success=False,
                            terminal_state='failed',
                            message=(
                                'Safety intervention timed out '
                                'without recovery.'
                            ),
                        )

                if execution_outcome == 'cancelled':
                    goal_handle.canceled()

                    return self._result(
                        success=False,
                        terminal_state='cancelled',
                        message=(
                            'Navigation execution was cancelled.'
                        ),
                    )

                if execution_outcome == 'controller_rejected':
                    goal_handle.abort()

                    return self._result(
                        success=False,
                        terminal_state='failed',
                        message=(
                            'JointTrajectoryController rejected '
                            'the generated trajectory.'
                        ),
                    )

                if execution_outcome != 'completed':
                    goal_handle.abort()

                    return self._result(
                        success=False,
                        terminal_state='failed',
                        message=(
                            'Unexpected trajectory execution outcome.'
                        ),
                    )

                if (
                    controller_result.error_code
                    != FollowJointTrajectory
                    .Result
                    .SUCCESSFUL
                ):
                    goal_handle.abort()

                    message = (
                        'JointTrajectoryController failed: '
                        f'error_code='
                        f'{controller_result.error_code}, '
                        f'error_string='
                        f'{controller_result.error_string}'
                    )

                    self.get_logger().error(
                        message
                    )

                    return self._result(
                        success=False,
                        terminal_state='failed',
                        message=message,
                    )

                self._publish_feedback(
                    goal_handle,
                    progress=1.0,
                    state='completed',
                )

                goal_handle.succeed()

                self.get_logger().info(
                    'Supervised navigation trajectory '
                    'completed successfully.'
                )

                return self._result(
                    success=True,
                    terminal_state='completed',
                    message=(
                        'Phase 2 trajectory executed successfully '
                        'under ROS autonomous supervision.'
                    ),
                )

        except Exception as error:
            self.get_logger().error(
                'Unexpected navigation execution failure: '
                f'{type(error).__name__}: {error}'
            )

            if goal_handle.is_active:
                goal_handle.abort()

            return self._result(
                success=False,
                terminal_state='failed',
                message=(
                    f'{type(error).__name__}: {error}'
                ),
            )

        finally:
            with self._lock:
                self._execution_active = False

            self._clear_command_state()


def main(
    args=None,
) -> None:
    """Run the supervised navigation execution node."""
    rclpy.init(
        args=args
    )

    node = (
        NavigationExecutionNode()
    )

    executor = (
        MultiThreadedExecutor(
            num_threads=4
        )
    )

    executor.add_node(
        node
    )

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
