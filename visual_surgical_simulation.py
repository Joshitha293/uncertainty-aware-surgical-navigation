"""Standalone 3D visualisation of uncertainty-aware surgical navigation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
)
from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.task_relevance import (
    SurgicalTask,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    ViewpointSamplingConfig,
    generate_candidate_viewpoints,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.planner import (
    plan_rrt,
)


def create_surgical_workspace():
    """Create the surgical planning workspace."""

    instrument = SurgicalInstrument(
        rcm_position=np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        )
    )

    structures = (
        SphericalStructure(
            centre=np.array(
                [0.14, 0.04, 0.0],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.array(
                [0.18, -0.06, 0.02],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )

    start_q = np.array(
        [0.0, 0.0, 0.10, 0.0],
        dtype=float,
    )

    goal_q = np.array(
        [
            0.6108652382,
            0.4363323120,
            0.25,
            0.0,
        ],
        dtype=float,
    )

    return (
        instrument,
        structures,
        start_q,
        goal_q,
    )


def sphere_surface(
    centre: np.ndarray,
    radius: float,
    resolution: int = 30,
):
    """Generate coordinates for a sphere."""

    u = np.linspace(
        0.0,
        2.0 * np.pi,
        resolution,
    )

    v = np.linspace(
        0.0,
        np.pi,
        resolution,
    )

    x = (
        radius
        * np.outer(
            np.cos(u),
            np.sin(v),
        )
        + centre[0]
    )

    y = (
        radius
        * np.outer(
            np.sin(u),
            np.sin(v),
        )
        + centre[1]
    )

    z = (
        radius
        * np.outer(
            np.ones_like(u),
            np.cos(v),
        )
        + centre[2]
    )

    return x, y, z


def path_to_tip_positions(
    instrument: SurgicalInstrument,
    path: np.ndarray,
) -> np.ndarray:
    """Convert the RRT joint-space path to Cartesian tip positions."""

    return np.vstack(
        [
            instrument.forward_position(q)
            for q in path
        ]
    )


def create_task(
    tip_positions: np.ndarray,
    structures: tuple[SphericalStructure, ...],
) -> SurgicalTask:
    """Create the trajectory-dependent surgical task."""

    critical_points = np.vstack(
        [
            structure.centre
            for structure in structures
        ]
    )

    return SurgicalTask(
        trajectory=tip_positions,
        safety_critical_points=critical_points,
    )


def create_active_perception(
    task: SurgicalTask,
    target: SphericalStructure,
):
    """Build and run the existing task-aware active-perception system."""

    camera = SurgicalCamera(
        intrinsics=CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(50.0),
            near_distance=0.05,
            far_distance=0.50,
        )
    )

    observation_model = ViewpointObservationModel(
        camera=camera,
    )

    generic_scorer = GenericViewpointScorer(
        observation_model=observation_model,
    )

    task_aware_scorer = TaskAwareViewpointScorer(
        generic_scorer=generic_scorer,
        task=task,
        task_config=TaskAwareScoringConfig(
            task_weight=2.0,
            alignment_weight=1.0,
            uncertainty_weight=1.0,
        ),
    )

    controller = TaskAwareActivePerception(
        scorer=task_aware_scorer,
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
        config=ViewpointSamplingConfig(
            radii=(
                0.15,
                0.20,
                0.25,
            ),
            azimuth_count=12,
            elevation_angles=(
                np.deg2rad(-20.0),
                0.0,
                np.deg2rad(20.0),
            ),
        ),
    )

    current_pose = CameraPose(
        position=np.array(
            [0.0, -0.25, 0.15],
            dtype=float,
        ),
        rotation=np.eye(3),
    )

    selection = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        occluders=(),
    )

    return (
        candidates,
        selection,
    )


def plot_workspace(
    instrument: SurgicalInstrument,
    structures: tuple[SphericalStructure, ...],
    path: np.ndarray,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    candidates,
    selection,
) -> None:
    """Create the integrated 3D visualisation."""

    figure = plt.figure(
        figsize=(13, 9)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    # ==================================================
    # ANATOMICAL STRUCTURES
    # ==================================================

    for index, structure in enumerate(
        structures,
        start=1,
    ):
        physical_x, physical_y, physical_z = (
            sphere_surface(
                structure.centre,
                structure.physical_radius,
            )
        )

        axis.plot_surface(
            physical_x,
            physical_y,
            physical_z,
            alpha=0.80,
        )

        safety_x, safety_y, safety_z = (
            sphere_surface(
                structure.centre,
                structure.safety_radius,
            )
        )

        axis.plot_wireframe(
            safety_x,
            safety_y,
            safety_z,
            alpha=0.20,
            linewidth=0.5,
        )

        axis.text(
            structure.centre[0],
            structure.centre[1],
            structure.centre[2]
            + structure.safety_radius
            + 0.01,
            f"Structure {index}",
        )

    # ==================================================
    # RCM
    # ==================================================

    rcm = instrument.rcm_position

    axis.scatter(
        rcm[0],
        rcm[1],
        rcm[2],
        s=120,
        marker="o",
        label="RCM",
    )

    # ==================================================
    # RRT TRAJECTORY
    # ==================================================

    tip_positions = path_to_tip_positions(
        instrument,
        path,
    )

    axis.plot(
        tip_positions[:, 0],
        tip_positions[:, 1],
        tip_positions[:, 2],
        linewidth=3.0,
        label="RRT trajectory",
    )

    # ==================================================
    # START / GOAL
    # ==================================================

    start_position = instrument.forward_position(
        start_q
    )

    goal_position = instrument.forward_position(
        goal_q
    )

    axis.scatter(
        start_position[0],
        start_position[1],
        start_position[2],
        s=100,
        marker="^",
        label="Start",
    )

    axis.scatter(
        goal_position[0],
        goal_position[1],
        goal_position[2],
        s=140,
        marker="*",
        label="Goal",
    )

    # ==================================================
    # FINAL INSTRUMENT
    # ==================================================

    proximal_point, tip_point = (
        instrument.shaft_segment(
            path[-1],
            proximal_length=0.10,
        )
    )

    axis.plot(
        [
            proximal_point[0],
            tip_point[0],
        ],
        [
            proximal_point[1],
            tip_point[1],
        ],
        [
            proximal_point[2],
            tip_point[2],
        ],
        linewidth=5.0,
        label="Instrument",
    )

    # ==================================================
    # CANDIDATE VIEWPOINTS
    # ==================================================

    for candidate in candidates:
        position = np.asarray(
            candidate.pose.position,
            dtype=float,
        )

        axis.scatter(
            position[0],
            position[1],
            position[2],
            s=18,
            alpha=0.30,
        )

    # ==================================================
    # SELECTED VIEWPOINT
    # ==================================================

    selected_position = np.asarray(
        selection.selected_position,
        dtype=float,
    )

    axis.scatter(
        selected_position[0],
        selected_position[1],
        selected_position[2],
        s=250,
        marker="*",
        label="Selected task-aware viewpoint",
    )

    target = structures[0].centre

    axis.plot(
        [
            selected_position[0],
            target[0],
        ],
        [
            selected_position[1],
            target[1],
        ],
        [
            selected_position[2],
            target[2],
        ],
        linestyle="--",
        linewidth=1.5,
    )

    # ==================================================
    # SELECTED VIEWPOINT PARAMETERS
    # ==================================================

    selected_viewpoint = (
        selection.selected_viewpoint
    )

    radius_mm = (
        selected_viewpoint.radius * 1000.0
    )

    azimuth_deg = np.rad2deg(
        selected_viewpoint.azimuth
    )

    elevation_deg = np.rad2deg(
        selected_viewpoint.elevation
    )

    uncertainty_mm = (
        selection.task_uncertainty * 1000.0
    )

    # ==================================================
    # RESEARCH METRICS PANEL
    # ==================================================

    metrics = (
        "TASK-AWARE ACTIVE PERCEPTION\n"
        "\n"
        f"Candidates evaluated: "
        f"{selection.candidate_count}\n"
        "\n"
        f"Task relevance: "
        f"{selection.task_relevance:.3f}\n"
        f"Task alignment: "
        f"{selection.task_alignment:.3f}\n"
        f"Task uncertainty: "
        f"{uncertainty_mm:.3f} mm\n"
        "\n"
        "SELECTED VIEWPOINT\n"
        f"Radius: "
        f"{radius_mm:.1f} mm\n"
        f"Azimuth: "
        f"{azimuth_deg:.1f}°\n"
        f"Elevation: "
        f"{elevation_deg:.1f}°\n"
        "\n"
        f"Raw task-aware utility: "
        f"{selection.selected_score.score:.3f}"
    )

    axis.text2D(
        0.02,
        0.97,
        metrics,
        transform=axis.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            alpha=0.88,
        ),
    )

    # ==================================================
    # AXIS / TITLE
    # ==================================================

    axis.set_xlabel(
        "X position (m)"
    )

    axis.set_ylabel(
        "Y position (m)"
    )

    axis.set_zlabel(
        "Z position (m)"
    )

    axis.set_title(
        "Uncertainty-Aware Surgical Navigation\n"
        "RRT + Task-Aware Active Perception"
    )

    axis.legend()

    axis.set_xlim(
        -0.15,
        0.35,
    )

    axis.set_ylim(
        -0.25,
        0.25,
    )

    axis.set_zlim(
        -0.20,
        0.25,
    )

    axis.set_box_aspect(
        (1.4, 1.0, 0.9)
    )

    figure.tight_layout()

    plt.show()


def main() -> None:
    """Run the integrated surgical-navigation simulation."""

    print()
    print("=" * 65)
    print(" UNCERTAINTY-AWARE SURGICAL NAVIGATION")
    print(" RRT + TASK-AWARE ACTIVE PERCEPTION")
    print("=" * 65)
    print()

    # ==================================================
    # 1. WORKSPACE
    # ==================================================

    (
        instrument,
        structures,
        start_q,
        goal_q,
    ) = create_surgical_workspace()

    # ==================================================
    # 2. RRT
    # ==================================================

    print(
        "1. Running collision-aware RRT..."
    )

    result = plan_rrt(
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        structures=structures,
        instrument_radius=0.006,
        proximal_length=0.10,
        max_iterations=5000,
        step_size=0.08,
        goal_bias=0.15,
        edge_resolution=20,
        seed=20000,
    )

    if not result.success:
        print(
            "ERROR: RRT failed to find a safe path."
        )
        return

    print(
        f"   Success: {result.success}"
    )

    print(
        f"   Iterations: {result.iterations}"
    )

    print(
        f"   Waypoints: {len(result.path)}"
    )

    # ==================================================
    # 3. CARTESIAN TRAJECTORY
    # ==================================================

    tip_positions = path_to_tip_positions(
        instrument,
        result.path,
    )

    # ==================================================
    # 4. SURGICAL TASK
    # ==================================================

    print()
    print(
        "2. Building trajectory-dependent surgical task..."
    )

    task = create_task(
        tip_positions=tip_positions,
        structures=structures,
    )

    print(
        f"   Trajectory points: "
        f"{len(task.trajectory)}"
    )

    print(
        f"   Safety-critical points: "
        f"{len(task.safety_critical_points)}"
    )

    # ==================================================
    # 5. ACTIVE PERCEPTION
    # ==================================================

    print()
    print(
        "3. Running task-aware active perception..."
    )

    candidates, selection = (
        create_active_perception(
            task=task,
            target=structures[0],
        )
    )

    print(
        f"   Candidates evaluated: "
        f"{selection.candidate_count}"
    )

    print(
        f"   Selected position: "
        f"{selection.selected_position}"
    )

    print(
        f"   Task relevance: "
        f"{selection.task_relevance:.4f}"
    )

    print(
        f"   Task alignment: "
        f"{selection.task_alignment:.4f}"
    )

    print(
        f"   Task uncertainty: "
        f"{selection.task_uncertainty * 1000.0:.3f} mm"
    )

    print(
        f"   Task-aware utility: "
        f"{selection.selected_score.score:.4f}"
    )

    print()
    print(
        "4. Opening integrated 3D simulation..."
    )

    plot_workspace(
        instrument=instrument,
        structures=structures,
        path=result.path,
        start_q=start_q,
        goal_q=goal_q,
        candidates=candidates,
        selection=selection,
    )


if __name__ == "__main__":
    main()