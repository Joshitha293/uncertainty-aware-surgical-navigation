"""OpenCV image-processing pipeline for Phase 3 surgical perception.

This module introduces genuine pixel-domain perception into the project.

The pipeline operates on simulated endoscopic-style BGR images and provides:

- image validation;
- Gaussian filtering;
- BGR-to-HSV colour conversion;
- HSV threshold segmentation;
- morphological mask cleanup;
- Canny edge extraction;
- contour detection;
- 2-D target centroid estimation;
- deterministic synthetic high-contrast marker rendering.

The synthetic marker is an engineering test target. Results from this module
must not be interpreted as clinical tissue-segmentation performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVRange:
    """Inclusive HSV threshold range using OpenCV conventions."""

    lower: tuple[int, int, int] = (
        35,
        80,
        60,
    )

    upper: tuple[int, int, int] = (
        85,
        255,
        255,
    )

    def __post_init__(self) -> None:
        if (
            len(self.lower) != 3
            or len(self.upper) != 3
        ):
            raise ValueError(
                "HSV bounds must contain three values."
            )

        limits = (
            179,
            255,
            255,
        )

        for index, (
            lower,
            upper,
            maximum,
        ) in enumerate(
            zip(
                self.lower,
                self.upper,
                limits,
            )
        ):
            if not (
                0 <= lower <= maximum
            ):
                raise ValueError(
                    f"HSV lower bound {index} is invalid."
                )

            if not (
                0 <= upper <= maximum
            ):
                raise ValueError(
                    f"HSV upper bound {index} is invalid."
                )

            if lower > upper:
                raise ValueError(
                    "HSV lower bounds must not exceed upper bounds."
                )


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters for classical target segmentation."""

    hsv_range: HSVRange = HSVRange()

    gaussian_kernel: int = 5

    morphology_kernel: int = 5

    opening_iterations: int = 1

    closing_iterations: int = 1

    canny_low_threshold: int = 50

    canny_high_threshold: int = 150

    minimum_contour_area: float = 50.0

    def __post_init__(self) -> None:
        if (
            self.gaussian_kernel < 1
            or self.gaussian_kernel % 2 == 0
        ):
            raise ValueError(
                "gaussian_kernel must be a positive odd integer."
            )

        if (
            self.morphology_kernel < 1
            or self.morphology_kernel % 2 == 0
        ):
            raise ValueError(
                "morphology_kernel must be a positive odd integer."
            )

        if self.opening_iterations < 0:
            raise ValueError(
                "opening_iterations must be non-negative."
            )

        if self.closing_iterations < 0:
            raise ValueError(
                "closing_iterations must be non-negative."
            )

        if not (
            0
            <= self.canny_low_threshold
            < self.canny_high_threshold
            <= 255
        ):
            raise ValueError(
                "Canny thresholds must satisfy 0 <= low < high <= 255."
            )

        if self.minimum_contour_area <= 0.0:
            raise ValueError(
                "minimum_contour_area must be positive."
            )


@dataclass(frozen=True)
class ContourDetection:
    """Largest valid segmented target."""

    found: bool

    centroid: np.ndarray | None

    area: float

    bounding_box: (
        tuple[
            int,
            int,
            int,
            int,
        ]
        | None
    )


@dataclass(frozen=True)
class SegmentationResult:
    """Intermediate and final outputs from target segmentation."""

    hsv_image: np.ndarray

    raw_mask: np.ndarray

    cleaned_mask: np.ndarray

    edges: np.ndarray

    detection: ContourDetection


def validate_bgr_image(
    image: np.ndarray,
) -> np.ndarray:
    """Validate an OpenCV BGR image."""

    image = np.asarray(
        image
    )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "image must have shape (height, width, 3)."
        )

    if (
        image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise ValueError(
            "image dimensions must be non-zero."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image must use uint8 pixel values."
        )

    return image


def gaussian_filter_bgr(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply Gaussian spatial filtering to a BGR image."""

    image = validate_bgr_image(
        image
    )

    if (
        kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "kernel_size must be a positive odd integer."
        )

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size,
        ),
        sigmaX=0.0,
    )


def bgr_to_hsv(
    image: np.ndarray,
) -> np.ndarray:
    """Convert BGR image representation to HSV."""

    image = validate_bgr_image(
        image
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )


def threshold_hsv(
    hsv_image: np.ndarray,
    hsv_range: HSVRange,
) -> np.ndarray:
    """Threshold an HSV image into a binary segmentation mask."""

    hsv_image = np.asarray(
        hsv_image
    )

    if (
        hsv_image.ndim != 3
        or hsv_image.shape[2] != 3
        or hsv_image.dtype != np.uint8
    ):
        raise ValueError(
            "hsv_image must be a uint8 image with shape (H, W, 3)."
        )

    lower = np.asarray(
        hsv_range.lower,
        dtype=np.uint8,
    )

    upper = np.asarray(
        hsv_range.upper,
        dtype=np.uint8,
    )

    return cv2.inRange(
        hsv_image,
        lower,
        upper,
    )


def clean_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: int = 5,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
) -> np.ndarray:
    """Remove isolated noise and close small mask holes."""

    mask = np.asarray(
        mask
    )

    if (
        mask.ndim != 2
        or mask.dtype != np.uint8
    ):
        raise ValueError(
            "mask must be a 2-D uint8 image."
        )

    if (
        kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "kernel_size must be a positive odd integer."
        )

    if opening_iterations < 0:
        raise ValueError(
            "opening_iterations must be non-negative."
        )

    if closing_iterations < 0:
        raise ValueError(
            "closing_iterations must be non-negative."
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    cleaned = mask.copy()

    if opening_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=opening_iterations,
        )

    if closing_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=closing_iterations,
        )

    return cleaned


def canny_edges(
    mask: np.ndarray,
    *,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Extract target boundaries using the Canny edge detector."""

    mask = np.asarray(
        mask
    )

    if (
        mask.ndim != 2
        or mask.dtype != np.uint8
    ):
        raise ValueError(
            "mask must be a 2-D uint8 image."
        )

    if not (
        0
        <= low_threshold
        < high_threshold
        <= 255
    ):
        raise ValueError(
            "Canny thresholds must satisfy 0 <= low < high <= 255."
        )

    return cv2.Canny(
        mask,
        low_threshold,
        high_threshold,
    )


def detect_largest_contour(
    mask: np.ndarray,
    *,
    minimum_area: float = 50.0,
) -> ContourDetection:
    """Locate the largest sufficiently large segmented contour."""

    mask = np.asarray(
        mask
    )

    if (
        mask.ndim != 2
        or mask.dtype != np.uint8
    ):
        raise ValueError(
            "mask must be a 2-D uint8 image."
        )

    if minimum_area <= 0.0:
        raise ValueError(
            "minimum_area must be positive."
        )

    contours, _hierarchy = (
        cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    )

    if not contours:
        return ContourDetection(
            found=False,
            centroid=None,
            area=0.0,
            bounding_box=None,
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(
            contour
        )
    )

    if area < minimum_area:
        return ContourDetection(
            found=False,
            centroid=None,
            area=area,
            bounding_box=None,
        )

    moments = cv2.moments(
        contour
    )

    if abs(
        moments["m00"]
    ) <= 1e-12:
        return ContourDetection(
            found=False,
            centroid=None,
            area=area,
            bounding_box=None,
        )

    centroid = np.asarray(
        [
            moments["m10"]
            / moments["m00"],
            moments["m01"]
            / moments["m00"],
        ],
        dtype=float,
    )

    x, y, width, height = (
        cv2.boundingRect(
            contour
        )
    )

    return ContourDetection(
        found=True,
        centroid=centroid,
        area=area,
        bounding_box=(
            int(
                x
            ),
            int(
                y
            ),
            int(
                width
            ),
            int(
                height
            ),
        ),
    )


def segment_target_bgr(
    image: np.ndarray,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Run the full classical target-segmentation pipeline."""

    if config is None:
        config = (
            SegmentationConfig()
        )

    filtered = (
        gaussian_filter_bgr(
            image,
            kernel_size=(
                config.gaussian_kernel
            ),
        )
    )

    hsv = bgr_to_hsv(
        filtered
    )

    raw_mask = threshold_hsv(
        hsv,
        config.hsv_range,
    )

    cleaned_mask = (
        clean_binary_mask(
            raw_mask,
            kernel_size=(
                config.morphology_kernel
            ),
            opening_iterations=(
                config.opening_iterations
            ),
            closing_iterations=(
                config.closing_iterations
            ),
        )
    )

    edges = canny_edges(
        cleaned_mask,
        low_threshold=(
            config.canny_low_threshold
        ),
        high_threshold=(
            config.canny_high_threshold
        ),
    )

    detection = (
        detect_largest_contour(
            cleaned_mask,
            minimum_area=(
                config.minimum_contour_area
            ),
        )
    )

    return SegmentationResult(
        hsv_image=hsv,
        raw_mask=raw_mask,
        cleaned_mask=(
            cleaned_mask
        ),
        edges=edges,
        detection=detection,
    )


def render_synthetic_marker_frame(
    *,
    width: int,
    height: int,
    marker_pixel: np.ndarray,
    marker_radius: int = 14,
    seed: int = 7,
) -> np.ndarray:
    """Render a deterministic endoscopic-style test image.

    The frame contains a reddish background, mild image noise,
    a simulated specular highlight and a green high-contrast navigation
    marker.

    This is an engineering validation image rather than simulated
    patient anatomy.
    """

    if width < 32:
        raise ValueError(
            "width must be at least 32."
        )

    if height < 32:
        raise ValueError(
            "height must be at least 32."
        )

    if marker_radius < 2:
        raise ValueError(
            "marker_radius must be at least 2 pixels."
        )

    marker_pixel = np.asarray(
        marker_pixel,
        dtype=float,
    )

    if marker_pixel.shape != (2,):
        raise ValueError(
            "marker_pixel must have shape (2,)."
        )

    marker_x = int(
        round(
            float(
                marker_pixel[0]
            )
        )
    )

    marker_y = int(
        round(
            float(
                marker_pixel[1]
            )
        )
    )

    if not (
        0 <= marker_x < width
        and 0 <= marker_y < height
    ):
        raise ValueError(
            "marker_pixel must lie inside the image."
        )

    image = np.empty(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # Muted reddish simulated endoscopic background.
    image[:] = np.asarray(
        [
            70,
            75,
            130,
        ],
        dtype=np.uint8,
    )

    # Add a darker peripheral vignette.
    vignette = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        vignette,
        (
            width // 2,
            height // 2,
        ),
        int(
            0.48
            * min(
                width,
                height,
            )
        ),
        255,
        thickness=-1,
    )

    darkened = (
        image.astype(
            np.float32
        )
        * 0.55
    ).astype(
        np.uint8
    )

    image = np.where(
        vignette[
            :,
            :,
            None,
        ]
        > 0,
        image,
        darkened,
    ).astype(
        np.uint8
    )

    # High-contrast simulated navigation marker.
    cv2.circle(
        image,
        (
            marker_x,
            marker_y,
        ),
        marker_radius,
        (
            40,
            220,
            40,
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Small non-target specular highlight.
    cv2.circle(
        image,
        (
            int(
                0.72
                * width
            ),
            int(
                0.28
                * height
            ),
        ),
        5,
        (
            245,
            245,
            245,
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(
        loc=0.0,
        scale=3.0,
        size=image.shape,
    )

    noisy = np.clip(
        image.astype(
            np.float32
        )
        + noise,
        0.0,
        255.0,
    )

    return noisy.astype(
        np.uint8
    )