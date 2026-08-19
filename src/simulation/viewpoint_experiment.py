"""Compare poor and improved camera viewpoints for surgical perception."""

from __future__ import annotations

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)


def make_camera() -> SurgicalCamera:
    """Create the simulated surgical camera."""

    return SurgicalCamera(
        CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.60,
        )
    )


def make_observation_model(
    camera: SurgicalCamera,
) -> ViewpointObservationModel:
    """Create the viewpoint-dependent perception model."""

    return ViewpointObservationModel(
        camera=camera,
        config=ObservationModelConfig(
            base_sigma=0.002,
            reference_distance=0.15,
            distance_weight=1.0,
            angle_weight=1.0,
            invisible_sigma=0.050,
            occluded_sigma=0.030,
        ),
    )


def make_target_structure() -> SphericalStructure:
    """Create the safety-critical anatomical target."""

    return SphericalStructure(
        centre=np.array(
            [0.24, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_occluder() -> SphericalStructure:
    """Create an anatomical structure that blocks the initial view."""

    return SphericalStructure(
        centre=np.array(
            [0.12, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )


def make_bad_viewpoint(
    target: SphericalStructure,
) -> CameraPose:
    """Create a camera pose whose line of sight is occluded."""

    camera_position = np.array(
        [0.00, 0.00, 0.00],
        dtype=float,
    )

    return CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target.centre,
        ),
    )


def make_improved_viewpoint(
    target: SphericalStructure,
) -> CameraPose:
    """Create a displaced pose with an unobstructed target view."""

    camera_position = np.array(
        [0.05, 0.13, 0.06],
        dtype=float,
    )

    return CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target.centre,
        ),
    )


def main() -> None:
    """Run a controlled poor-view versus improved-view experiment."""

    camera = make_camera()

    model = make_observation_model(
        camera
    )

    target = make_target_structure()

    occluder = make_occluder()

    bad_pose = make_bad_viewpoint(
        target
    )

    improved_pose = make_improved_viewpoint(
        target
    )

    bad_quality = model.observation_quality(
        camera_pose=bad_pose,
        structure=target,
        occluders=(occluder,),
    )

    improved_quality = model.observation_quality(
        camera_pose=improved_pose,
        structure=target,
        occluders=(occluder,),
    )

    trial_count = 1000

    bad_errors: list[float] = []
    improved_errors: list[float] = []

    for trial in range(
        trial_count
    ):
        seed = 10000 + trial

        bad_observation = (
            model.observe_structure(
                camera_pose=bad_pose,
                structure=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=(occluder,),
            )
        )

        improved_observation = (
            model.observe_structure(
                camera_pose=improved_pose,
                structure=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=(occluder,),
            )
        )

        bad_errors.append(
            bad_observation.localisation_error
        )

        improved_errors.append(
            improved_observation.localisation_error
        )

    bad_errors_array = np.asarray(
        bad_errors,
        dtype=float,
    )

    improved_errors_array = np.asarray(
        improved_errors,
        dtype=float,
    )

    mean_bad_error = float(
        np.mean(
            bad_errors_array
        )
    )

    mean_improved_error = float(
        np.mean(
            improved_errors_array
        )
    )

    median_bad_error = float(
        np.median(
            bad_errors_array
        )
    )

    median_improved_error = float(
        np.median(
            improved_errors_array
        )
    )

    error_reduction = (
        100.0
        * (
            mean_bad_error
            - mean_improved_error
        )
        / mean_bad_error
    )

    sigma_reduction = (
        100.0
        * (
            bad_quality.localisation_sigma
            - improved_quality.localisation_sigma
        )
        / bad_quality.localisation_sigma
    )

    camera_displacement = float(
        np.linalg.norm(
            improved_pose.position
            - bad_pose.position
        )
    )

    improved_better_fraction = float(
        np.mean(
            improved_errors_array
            < bad_errors_array
        )
    )

    print()
    print("Viewpoint-dependent perception experiment")
    print("-----------------------------------------")

    print()
    print("Poor initial viewpoint")
    print("----------------------")
    print(
        f"Visible in frustum: "
        f"{bad_quality.visible}"
    )
    print(
        f"Occluded: "
        f"{bad_quality.occluded}"
    )
    print(
        "Predicted localisation sigma: "
        f"{bad_quality.localisation_sigma:.6f} m"
    )
    print(
        f"Camera-target distance: "
        f"{bad_quality.distance:.6f} m"
    )

    print()
    print("Improved viewpoint")
    print("------------------")
    print(
        f"Visible in frustum: "
        f"{improved_quality.visible}"
    )
    print(
        f"Occluded: "
        f"{improved_quality.occluded}"
    )
    print(
        "Predicted localisation sigma: "
        f"{improved_quality.localisation_sigma:.6f} m"
    )
    print(
        f"Camera-target distance: "
        f"{improved_quality.distance:.6f} m"
    )

    print()
    print("Matched observation experiment")
    print("------------------------------")
    print(
        f"Trials: "
        f"{trial_count}"
    )
    print(
        "Mean localisation error, poor view: "
        f"{mean_bad_error:.6f} m"
    )
    print(
        "Mean localisation error, improved view: "
        f"{mean_improved_error:.6f} m"
    )
    print(
        "Median localisation error, poor view: "
        f"{median_bad_error:.6f} m"
    )
    print(
        "Median localisation error, improved view: "
        f"{median_improved_error:.6f} m"
    )
    print(
        "Mean localisation-error reduction: "
        f"{error_reduction:.2f}%"
    )
    print(
        "Predicted sigma reduction: "
        f"{sigma_reduction:.2f}%"
    )
    print(
        "Improved viewpoint lower error in: "
        f"{100.0 * improved_better_fraction:.1f}% "
        f"of matched trials"
    )
    print(
        "Camera displacement required: "
        f"{camera_displacement:.6f} m"
    )


if __name__ == "__main__":
    main()