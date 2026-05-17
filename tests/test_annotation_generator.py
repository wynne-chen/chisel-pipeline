"""Tests for libs/annotation_generator.py.

Group 6 (mask_to_polygons_with_holes) fails RED against the current code,
exposing TWO bugs — both visible only once tests exist:

BUG A — annotation_generator.py:92 (proximate cause of every Group 6 failure)

    if not contours or not hierarchy:
        return []

When `cv2.findContours` finds at least one contour, `hierarchy` is a
non-empty np.ndarray. `not <ndarray>` invokes ndarray.__bool__ which
raises `ValueError: The truth value of an array with more than one
element is ambiguous`. The intent was clearly `hierarchy is None`
(cv2 returns None when contours is empty). The empty-mask test passes
because `not contours` short-circuits before the bad expression runs.

BUG B — annotation_generator.py:110-111 (downstream, currently masked)

    for i, contour in enumerate(hierarchy):
        parent_idx = hierarchy[3]

The loop variable `contour` here is actually a hierarchy row (a 4-element
array `[next, prev, first_child, parent]`), not a contour. `hierarchy[3]`
returns the 4th hierarchy row instead of the current row's parent field.
The correct expression is `contour[3]`. This bug is masked today by Bug A
— the function never reaches line 111 — but will surface as soon as Bug A
is fixed.

The tests below describe the *correct* behaviour and will turn GREEN once
both bugs are fixed.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from libs.annotation_generator import AnnotationGenerator


# A flat YOLO-style polygon is [x0, y0, x1, y1, ...]. Most tests work with
# arrays of (x, y) pairs reshaped from that flat list.
def _coords(flat_poly):
    return np.array(flat_poly).reshape(-1, 2)


def _cv_contour(points_xy):
    """Build an OpenCV-style contour of shape (N, 1, 2) from a list of (x, y)."""
    return np.array([[[x, y]] for x, y in points_xy], dtype=np.int32)


# ---------------------------------------------------------------------------
# Group 1 — is_clockwise
# ---------------------------------------------------------------------------


def test_is_clockwise_returns_true_for_clockwise_square_in_image_coords(gen):
    # TL -> TR -> BR -> BL is visually clockwise in image coords (y-down).
    # is_clockwise returns a numpy scalar (from `value < 0`), so check
    # truthiness rather than identity against Python True/False.
    clockwise_square = _cv_contour([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert gen.is_clockwise(clockwise_square)


def test_is_clockwise_returns_false_for_counterclockwise_square(gen):
    # TL -> BL -> BR -> TR is counter-clockwise in image coords.
    ccw_square = _cv_contour([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert not gen.is_clockwise(ccw_square)


# ---------------------------------------------------------------------------
# Group 2 — get_merge_pt_idx
# ---------------------------------------------------------------------------


def test_get_merge_pt_idx_returns_closest_pair_indices(gen):
    outer = _cv_contour([(0, 0), (100, 0), (100, 100), (0, 100)])
    inner = _cv_contour([(40, 40), (60, 40), (60, 60), (40, 60)])

    idx1, idx2 = gen.get_merge_pt_idx(outer, inner)

    outer_pt = outer[idx1][0]
    inner_pt = inner[idx2][0]
    chosen_dist = (outer_pt[0] - inner_pt[0]) ** 2 + (outer_pt[1] - inner_pt[1]) ** 2

    # No other pair across the two contours should be strictly closer.
    for op in outer:
        for ip in inner:
            d = (op[0][0] - ip[0][0]) ** 2 + (op[0][1] - ip[0][1]) ** 2
            assert d >= chosen_dist


# ---------------------------------------------------------------------------
# Group 3 — merge_contours
# ---------------------------------------------------------------------------


def test_merge_contours_returns_combined_point_count(gen):
    c1 = _cv_contour([(0, 0), (10, 0), (10, 10), (0, 10)])
    c2 = _cv_contour([(3, 3), (5, 3), (5, 5), (3, 5)])

    merged = gen.merge_contours(c1, c2, idx1=2, idx2=1)

    # Bridge pattern duplicates the merge points on both contours, so the
    # total point count is len(c1) + len(c2) + 2.
    assert merged.shape[0] == len(c1) + len(c2) + 2


def test_merge_contours_visits_both_inputs(gen):
    c1 = _cv_contour([(0, 0), (10, 0), (10, 10), (0, 10)])
    c2 = _cv_contour([(3, 3), (5, 3), (5, 5), (3, 5)])

    merged = gen.merge_contours(c1, c2, idx1=2, idx2=1)
    merged_pts = {tuple(p[0]) for p in merged}

    assert {(0, 0), (10, 0), (10, 10), (0, 10)} <= merged_pts
    assert {(3, 3), (5, 3), (5, 5), (3, 5)} <= merged_pts


# ---------------------------------------------------------------------------
# Group 4 — merge_with_parent
# ---------------------------------------------------------------------------


def test_merge_with_parent_runs_and_returns_combined_geometry(gen):
    parent_cw = _cv_contour([(0, 0), (100, 0), (100, 100), (0, 100)])
    child_ccw = _cv_contour([(40, 40), (40, 60), (60, 60), (60, 40)])

    merged = gen.merge_with_parent(parent_cw, child_ccw)

    assert merged.ndim == 3 and merged.shape[1:] == (1, 2)
    # The merged result must include points from both inputs (post any reversal).
    merged_pts = {tuple(p[0]) for p in merged}
    assert {(0, 0), (100, 100)} <= merged_pts
    assert {(40, 40), (60, 60)} <= merged_pts


# ---------------------------------------------------------------------------
# Group 5 — mask_to_polygons (no-holes path)
# ---------------------------------------------------------------------------


def test_mask_to_polygons_empty_mask_returns_empty_list(gen, empty_mask):
    assert gen.mask_to_polygons(empty_mask) == []


def test_mask_to_polygons_single_square_returns_one_polygon(gen, square_mask):
    polygons = gen.mask_to_polygons(square_mask)
    assert len(polygons) == 1


def test_mask_to_polygons_polygon_has_at_least_eight_coords(gen, square_mask):
    # A rectangle has 4 vertices = 8 coords. approxPolyDP may not reduce below this.
    polygons = gen.mask_to_polygons(square_mask)
    assert len(polygons[0]) >= 8


def test_mask_to_polygons_skips_sub_three_point_contours(gen, sub_min_contour_mask):
    # A single-pixel blob yields a contour with < 3 points; should be skipped.
    polygons = gen.mask_to_polygons(sub_min_contour_mask)
    assert polygons == []


def test_mask_to_polygons_two_disjoint_squares_returns_two_polygons(gen, two_squares_mask):
    polygons = gen.mask_to_polygons(two_squares_mask)
    assert len(polygons) == 2


def test_mask_to_polygons_three_dim_mask_is_squeezed(gen, square_mask):
    # (1, H, W) should be squeezed to 2D without error.
    mask_3d = square_mask[np.newaxis, :, :]
    polygons = gen.mask_to_polygons(mask_3d)
    assert len(polygons) == 1


def test_mask_to_polygons_raises_on_non_2d_mask(gen):
    bad_mask = np.zeros((3, 100, 100), dtype=np.uint8)  # cannot squeeze to 2D
    bad_mask[:, 10:90, 10:90] = 255
    with pytest.raises(ValueError, match="must be 2D"):
        gen.mask_to_polygons(bad_mask)


# ---------------------------------------------------------------------------
# Group 6 — mask_to_polygons_with_holes (CURRENTLY RED — exposes the bug)
# ---------------------------------------------------------------------------


def test_mask_with_holes_empty_returns_empty_list(gen, empty_mask):
    assert gen.mask_to_polygons_with_holes(empty_mask) == []


def test_mask_with_one_hole_returns_one_polygon(gen, donut_mask):
    """RED. Fails today at annotation_generator.py:92 with `ValueError:
    truth value of an array...` (Bug A in the module docstring). Bug B
    at line 111 is masked behind this guard and will surface once A is
    fixed.
    """
    polygons = gen.mask_to_polygons_with_holes(donut_mask)
    assert len(polygons) == 1


def test_mask_with_one_hole_polygon_visits_outer_and_inner_boundaries(gen, donut_mask):
    """RED. The merged polygon must touch both the outer square boundary and
    the inner hole boundary."""
    polygons = gen.mask_to_polygons_with_holes(donut_mask)
    assert len(polygons) == 1
    coords = _coords(polygons[0])
    xs, ys = coords[:, 0], coords[:, 1]

    # Outer extents: x and y both span roughly [10, 89].
    assert xs.min() <= 12 and xs.max() >= 87
    assert ys.min() <= 12 and ys.max() >= 87

    # Inner-hole vertices must appear (roughly in [40, 59]).
    inner_xs = coords[(xs >= 38) & (xs <= 61) & (ys >= 38) & (ys <= 61)]
    assert len(inner_xs) > 0, "merged polygon should include inner-hole points"


def test_mask_with_two_donuts_returns_two_polygons(gen, two_donuts_mask):
    """RED. Two disjoint donuts must produce two outer-ring polygons."""
    polygons = gen.mask_to_polygons_with_holes(two_donuts_mask)
    assert len(polygons) == 2


def test_mask_with_hole_below_min_area_skips_the_hole(gen, hole_below_min_area_mask):
    """RED. A 3x3 hole has area 9; with min_hole_area=10 it should be filtered,
    leaving only the outer ring with no inner-hole coords."""
    polygons = gen.mask_to_polygons_with_holes(hole_below_min_area_mask)
    assert len(polygons) == 1
    coords = _coords(polygons[0])
    xs, ys = coords[:, 0], coords[:, 1]
    interior = coords[(xs > 30) & (xs < 70) & (ys > 30) & (ys < 70)]
    assert len(interior) == 0, "filtered hole should not contribute vertices"


def test_mask_with_more_holes_than_max_skips_hole_merges(gen, many_holes_mask):
    """RED. When holes > max_holes, the parent contour is returned without
    any hole merges."""
    polygons = gen.mask_to_polygons_with_holes(many_holes_mask)
    assert len(polygons) == 1
    coords = _coords(polygons[0])
    xs, ys = coords[:, 0], coords[:, 1]
    # No point should sit inside the band of holes (y in ~[45,50]).
    in_hole_band = coords[(ys >= 44) & (ys <= 51) & (xs >= 18) & (xs <= 182)]
    assert len(in_hole_band) == 0


def test_mask_with_holes_three_dim_mask_is_squeezed(gen, donut_mask):
    """RED. (1, H, W) input should be squeezed before findContours."""
    mask_3d = donut_mask[np.newaxis, :, :]
    polygons = gen.mask_to_polygons_with_holes(mask_3d)
    assert len(polygons) == 1


def test_mask_with_holes_raises_on_non_2d_mask(gen):
    bad_mask = np.zeros((3, 100, 100), dtype=np.uint8)
    bad_mask[:, 10:90, 10:90] = 255
    with pytest.raises(ValueError, match="must be 2D"):
        gen.mask_to_polygons_with_holes(bad_mask)


# ---------------------------------------------------------------------------
# Group 7 — generate_tile_yolo_annotation
# ---------------------------------------------------------------------------


def test_generate_yolo_annotation_writes_class_zero_regardless_of_class_id(gen):
    polygons = [[0, 0, 100, 0, 100, 100, 0, 100]]
    lines = gen.generate_tile_yolo_annotation(polygons, class_id=42, img_w=100, img_h=100)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_generate_yolo_annotation_normalises_coords_to_zero_one(gen):
    polygons = [[0, 0, 100, 0, 100, 100, 0, 100]]
    lines = gen.generate_tile_yolo_annotation(polygons, class_id=0, img_w=100, img_h=100)
    tokens = lines[0].split()[1:]
    values = [float(t) for t in tokens]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_generate_yolo_annotation_skips_polygons_with_fewer_than_six_coords(gen):
    polygons = [[0, 0, 10, 0]]  # only 2 points = 4 coords
    lines = gen.generate_tile_yolo_annotation(polygons, class_id=0, img_w=100, img_h=100)
    assert lines == []


def test_generate_yolo_annotation_six_decimal_formatting(gen):
    polygons = [[0, 0, 50, 0, 50, 50, 0, 50]]
    lines = gen.generate_tile_yolo_annotation(polygons, class_id=0, img_w=100, img_h=100)
    pattern = re.compile(r"^\d+\.\d{6}$")
    for token in lines[0].split()[1:]:
        assert pattern.match(token), f"token {token!r} not formatted to 6 decimals"


def test_generate_yolo_annotation_empty_polygons_returns_empty_list(gen):
    assert gen.generate_tile_yolo_annotation([], class_id=0, img_w=100, img_h=100) == []


# ---------------------------------------------------------------------------
# Group 8 — create_coco_annotations
# ---------------------------------------------------------------------------


def test_create_coco_annotations_assigns_incrementing_ids(gen):
    polygons = [
        [0, 0, 10, 0, 10, 10, 0, 10],
        [20, 20, 30, 20, 30, 30, 20, 30],
    ]
    anns = gen.create_coco_annotations(image_id=7, polygons=polygons, category_id=3)
    assert [a["id"] for a in anns] == [1, 2]
    # next_annotation_id advances across calls.
    more = gen.create_coco_annotations(image_id=8, polygons=polygons[:1], category_id=3)
    assert more[0]["id"] == 3


def test_create_coco_annotations_returns_required_keys(gen):
    polygons = [[0, 0, 10, 0, 10, 10, 0, 10]]
    [ann] = gen.create_coco_annotations(image_id=1, polygons=polygons, category_id=5)
    required = {"id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"}
    assert required <= set(ann)
    assert ann["image_id"] == 1
    assert ann["category_id"] == 5
    assert ann["iscrowd"] == 0
    assert ann["segmentation"] == [polygons[0]]


def test_create_coco_annotations_area_matches_contour_area(gen):
    # A 10x10 axis-aligned square has area 100.
    polygons = [[0, 0, 10, 0, 10, 10, 0, 10]]
    [ann] = gen.create_coco_annotations(image_id=1, polygons=polygons, category_id=0)
    assert ann["area"] == pytest.approx(100.0)
    assert ann["bbox"] == [0, 0, 11, 11]  # cv2.boundingRect is inclusive of extents


# ---------------------------------------------------------------------------
# Group 9 — convert_coco_to_yolo (filesystem)
# ---------------------------------------------------------------------------


def test_convert_coco_to_yolo_writes_one_txt_per_annotated_image(gen, tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "img_a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "img_b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "segmentation": [[0, 0, 50, 0, 50, 50, 0, 50]]},
            {"id": 2, "image_id": 2, "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]]},
        ],
    }
    gen.convert_coco_to_yolo(coco, set_name="setX", output_dir=str(tmp_path))

    out_dir = tmp_path / "setX" / "labels"
    written = sorted(p.name for p in out_dir.glob("*.txt"))
    assert written == ["img_a.txt", "img_b.txt"]


def test_convert_coco_to_yolo_normalises_coords(gen, tmp_path):
    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]]},
        ],
    }
    gen.convert_coco_to_yolo(coco, set_name="s", output_dir=str(tmp_path))

    content = (tmp_path / "s" / "labels" / "img.txt").read_text().strip()
    tokens = content.split()
    assert tokens[0] == "0"
    values = [float(t) for t in tokens[1:]]
    assert values == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def test_convert_coco_to_yolo_skips_images_without_annotations(gen, tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "lonely.jpg", "width": 100, "height": 100},
        ],
        "annotations": [],
    }
    gen.convert_coco_to_yolo(coco, set_name="s", output_dir=str(tmp_path))

    out_dir = tmp_path / "s" / "labels"
    assert list(out_dir.glob("*.txt")) == []
