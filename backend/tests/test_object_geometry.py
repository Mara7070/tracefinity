"""Standalone outline contract tests; no app or persistence fixtures needed."""

import unittest
from dataclasses import FrozenInstanceError
from itertools import product

from app.services.object_geometry import CanonicalOutline

RECTANGLE = ((-2.0, -1.0), (4.0, -1.0), (4.0, 3.0), (-2.0, 3.0))


class CanonicalOutlineTests(unittest.TestCase):
    def test_equivalent_rings_have_identical_canonical_points(self):
        for closed, clockwise, start in product((False, True), (False, True), range(4)):
            with self.subTest(closed=closed, clockwise=clockwise, start=start):
                points = RECTANGLE[start:] + RECTANGLE[:start]
                if clockwise:
                    points = points[::-1]
                if closed:
                    points += points[:1]
                outline = CanonicalOutline.from_points(points)
                self.assertEqual(outline.points, RECTANGLE)
                self.assertEqual(outline.signed_area, 24.0)

    def test_consecutive_duplicates_including_closing_seam_are_removed(self):
        a, b, c, d = RECTANGLE
        outline = CanonicalOutline.from_points([a, a, b, b, c, d, d, a, a])
        self.assertEqual(outline.points, RECTANGLE)

    def test_bounds_and_dimensions(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        self.assertEqual(outline.bounds, (-2.0, -1.0, 4.0, 3.0))
        self.assertEqual(outline.width, 6.0)
        self.assertEqual(outline.height, 4.0)

    def test_triangle_area_and_unrounded_coordinates(self):
        points = ((0.123456789012345, 0.0), (4.0, 0.0), (0.123456789012345, 3.0))
        outline = CanonicalOutline.from_points(points)
        self.assertEqual(outline.points, points)
        self.assertAlmostEqual(outline.signed_area, (4.0 - points[0][0]) * 3.0 / 2)

    def test_too_few_distinct_vertices(self):
        for points in ([], [(0, 0)], [(0, 0), (1, 1)],
                       [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0)]):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "three distinct"):
                CanonicalOutline.from_points(points)

    def test_zero_area_is_rejected(self):
        for points in (((0, 0), (1, 1), (2, 2)),
                       ((0, 2), (1, 2), (3, 2)),
                       ((0, 0), (2, 2), (0, 2), (2, 0))):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, "nonzero area"):
                CanonicalOutline.from_points(points)

    def test_nonfinite_coordinates_are_rejected(self):
        for value, axis in product((float("nan"), float("inf"), -float("inf")), (0, 1)):
            point = [1.0, 1.0]
            point[axis] = value
            with self.subTest(value=value, axis=axis), self.assertRaisesRegex(ValueError, "finite"):
                CanonicalOutline.from_points([(0, 0), (2, 0), point])

    def test_points_are_immutable_float_tuples_and_detached_from_input(self):
        points = [[0, 0], [2, 0], [0, 2]]
        outline = CanonicalOutline.from_points(iter(points))
        points[0][0] = 99
        self.assertEqual(outline.points, ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)))
        self.assertIsInstance(outline.points, tuple)
        for point in outline.points:
            self.assertIsInstance(point, tuple)
            for value in point:
                self.assertIs(type(value), float)
        with self.assertRaises(TypeError):
            outline.points[0][0] = 1.0
        with self.assertRaises(FrozenInstanceError):
            outline.points = ()

    def test_repeated_minimum_has_deterministic_cyclic_order(self):
        points = ((0, 0), (2, 0), (1, 1), (0, 0), (1, 2), (0, 2))
        expected = CanonicalOutline.from_points(points)
        for start in range(len(points)):
            rotated = points[start:] + points[:start]
            for ring in (rotated, rotated[::-1]):
                with self.subTest(ring=ring):
                    self.assertEqual(CanonicalOutline.from_points(ring), expected)
        self.assertGreater(expected.signed_area, 0)


if __name__ == "__main__":
    unittest.main()
