"""Shared fixtures for the test suite.

All mask fixtures match the contract produced by
`ImageProcessor.filter_mask_by_class_id`: 2D, np.uint8, values in {0, 255}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable so `from libs....` works from tests/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from libs.annotation_generator import AnnotationGenerator  # noqa: E402


# ---- Config / generator ----------------------------------------------------


@pytest.fixture
def default_config() -> dict:
    """Config matching the production defaults, with min_hole_area low enough
    that the donut fixture's 400-pixel hole is retained.
    """
    return {
        "epsilon": 0.5,
        "epsilon_merge": 1.0,
        "min_hole_area": 10,
        "max_holes": 10,
        "merge_holes": True,
    }


@pytest.fixture
def gen(default_config) -> AnnotationGenerator:
    return AnnotationGenerator(config=default_config)


# ---- Mask fixtures ---------------------------------------------------------


@pytest.fixture
def empty_mask() -> np.ndarray:
    """All zeros — exercises the no-contours early return."""
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def square_mask() -> np.ndarray:
    """A single filled rectangle, no holes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    return mask


@pytest.fixture
def two_squares_mask() -> np.ndarray:
    """Two disjoint filled rectangles."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[10:90, 110:190] = 255
    return mask


@pytest.fixture
def donut_mask() -> np.ndarray:
    """A square with a single 20x20 hole — exercises the hole hierarchy path."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[40:60, 40:60] = 0
    return mask


@pytest.fixture
def two_donuts_mask() -> np.ndarray:
    """Two disjoint donuts — exercises multi-parent hierarchy."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[40:60, 40:60] = 0
    mask[10:90, 110:190] = 255
    mask[40:60, 140:160] = 0
    return mask


@pytest.fixture
def hole_below_min_area_mask() -> np.ndarray:
    """A square with a 3x3 hole (area=9). With min_hole_area=10 the hole is filtered."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    mask[50:53, 50:53] = 0
    return mask


@pytest.fixture
def many_holes_mask() -> np.ndarray:
    """A square with 11 small holes — exceeds default max_holes=10."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:90, 10:190] = 255
    # 11 disjoint 5x5 holes spaced along the interior
    for k in range(11):
        cx = 20 + k * 15
        mask[45:50, cx : cx + 5] = 0
    return mask


@pytest.fixture
def sub_min_contour_mask() -> np.ndarray:
    """A single isolated white pixel — produces a sub-3-point contour."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    return mask
