"""Tests for Phase 3 OpenCV image-processing pipeline."""

import cv2
import numpy as np
import pytest

from src.perception.camera import (
    CameraPose,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
    reconstruct_world_point_from_depth,
)
from src.perception.image_processing import (
    HSVRange,
    SegmentationConfig,
    bgr_to_hsv,
    canny_edges,
    clean_binary_mask,
    detect_largest_contour,
    gaussian_filter_bgr,
    render_synthetic_marker_frame,
    segment_target_bgr,
    threshold_hsv,
    validate_bgr_image,
)


def make_intrinsics() -> PixelCameraIntrinsics:
    return PixelCameraIntrinsics(
        width=320,
        height=240,
        fx=250.0,
        fy=250.0,
        cx=159.5,
        cy=119.5,
    )


def identity_pose() -> CameraPose:
    return CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=np.eye(
            3,
            dtype=float,
        ),
    )


def test_invalid_bgr_shape_is_rejected():
    with pytest.raises(
        ValueError,
        match="shape",
    ):
        validate_bgr_image(
            np.zeros(
                (
                    100,
                    100,
                ),
                dtype=np.uint8,
            )
        )


def test_gaussian_filter_preserves_image_shape():
    image = np.zeros(
        (
            80,
            120,
            3,
        ),
        dtype=np.uint8,
    )

    filtered = (
        gaussian_filter_bgr(
            image,
            kernel_size=5,
        )
    )

    assert (
        filtered.shape
        == image.shape
    )

    assert (
        filtered.dtype
        == np.uint8
    )


def test_bgr_to_hsv_preserves_dimensions():
    image = np.zeros(
        (
            60,
            90,
            3,
        ),
        dtype=np.uint8,
    )

    hsv = bgr_to_hsv(
        image
    )

    assert hsv.shape == image.shape
    assert hsv.dtype == np.uint8


def test_hsv_threshold_detects_green_marker():
    image = np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        image,
        (
            50,
            50,
        ),
        10,
        (
            40,
            220,
            40,
        ),
        thickness=-1,
    )

    mask = threshold_hsv(
        bgr_to_hsv(
            image
        ),
        HSVRange(),
    )

    assert (
        mask[
            50,
            50
        ]
        == 255
    )

    assert (
        mask[
            5,
            5
        ]
        == 0
    )


def test_morphological_opening_removes_isolated_noise():
    mask = np.zeros(
        (
            100,
            100,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            50,
            50,
        ),
        10,
        255,
        thickness=-1,
    )

    mask[
        5,
        5
    ] = 255

    cleaned = (
        clean_binary_mask(
            mask,
            kernel_size=5,
            opening_iterations=1,
            closing_iterations=0,
        )
    )

    assert (
        cleaned[
            5,
            5
        ]
        == 0
    )

    assert (
        cleaned[
            50,
            50
        ]
        == 255
    )


def test_canny_detects_segmented_boundary():
    mask = np.zeros(
        (
            100,
            100,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            50,
            50,
        ),
        15,
        255,
        thickness=-1,
    )

    edges = canny_edges(
        mask
    )

    assert (
        np.count_nonzero(
            edges
        )
        > 0
    )


def test_largest_contour_returns_correct_centroid():
    mask = np.zeros(
        (
            120,
            160,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            80,
            60,
        ),
        15,
        255,
        thickness=-1,
    )

    detection = (
        detect_largest_contour(
            mask,
            minimum_area=50.0,
        )
    )

    assert detection.found

    np.testing.assert_allclose(
        detection.centroid,
        np.asarray(
            [
                80.0,
                60.0,
            ]
        ),
        atol=0.25,
    )

    assert detection.area > 500.0


def test_empty_mask_returns_no_detection():
    mask = np.zeros(
        (
            100,
            100,
        ),
        dtype=np.uint8,
    )

    detection = (
        detect_largest_contour(
            mask
        )
    )

    assert not detection.found
    assert detection.centroid is None
    assert detection.area == 0.0


def test_small_contour_is_rejected_by_area():
    mask = np.zeros(
        (
            100,
            100,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            50,
            50,
        ),
        2,
        255,
        thickness=-1,
    )

    detection = (
        detect_largest_contour(
            mask,
            minimum_area=100.0,
        )
    )

    assert not detection.found


def test_full_pipeline_localises_synthetic_marker():
    expected_pixel = np.asarray(
        [
            145.0,
            92.0,
        ],
        dtype=float,
    )

    image = (
        render_synthetic_marker_frame(
            width=320,
            height=240,
            marker_pixel=expected_pixel,
            marker_radius=14,
            seed=7,
        )
    )

    result = (
        segment_target_bgr(
            image
        )
    )

    assert result.detection.found

    np.testing.assert_allclose(
        result.detection.centroid,
        expected_pixel,
        atol=1.0,
    )


def test_segmentation_masks_are_binary():
    image = (
        render_synthetic_marker_frame(
            width=320,
            height=240,
            marker_pixel=np.asarray(
                [
                    140.0,
                    110.0,
                ]
            ),
        )
    )

    result = (
        segment_target_bgr(
            image
        )
    )

    for mask in (
        result.raw_mask,
        result.cleaned_mask,
        result.edges,
    ):
        values = set(
            np.unique(
                mask
            ).tolist()
        )

        assert values.issubset(
            {
                0,
                255,
            }
        )


def test_image_detection_connects_to_3d_camera_geometry():
    """Image-derived centroid should recover the simulated 3-D target."""

    intrinsics = (
        make_intrinsics()
    )

    pose = identity_pose()

    target_world = np.asarray(
        [
            0.020,
            -0.010,
            0.250,
        ],
        dtype=float,
    )

    projection = (
        project_world_point(
            intrinsics,
            pose,
            target_world,
        )
    )

    assert projection.inside_image

    image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                projection.pixel
            ),
            marker_radius=14,
            seed=17,
        )
    )

    segmentation = (
        segment_target_bgr(
            image,
            config=(
                SegmentationConfig(
                    minimum_contour_area=100.0
                )
            ),
        )
    )

    assert (
        segmentation
        .detection
        .found
    )

    detected_pixel = (
        segmentation
        .detection
        .centroid
    )

    reconstructed_world = (
        reconstruct_world_point_from_depth(
            intrinsics,
            pose,
            detected_pixel,
            projection.depth,
        )
    )

    error = float(
        np.linalg.norm(
            reconstructed_world
            - target_world
        )
    )

    # Pixel quantisation and segmentation cause a small reconstruction error.
    assert error < 0.002