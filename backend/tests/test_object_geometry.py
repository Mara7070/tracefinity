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

    def test_translation_and_inverse(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        shifted = outline.translated(5, -7)
        self.assertEqual(shifted.points, ((3.0, -8.0), (9.0, -8.0),
                                          (9.0, -4.0), (3.0, -4.0)))
        self.assertEqual(shifted.bounds, (3.0, -8.0, 9.0, -4.0))
        self.assertEqual((shifted.width, shifted.height), (6.0, 4.0))
        self.assertEqual(shifted.signed_area, outline.signed_area)
        self.assertEqual(shifted.translated(-5, 7), outline)

    def test_exact_quarter_turns(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        expected = (
            RECTANGLE,
            ((-3.0, -2.0), (1.0, -2.0), (1.0, 4.0), (-3.0, 4.0)),
            ((-4.0, -3.0), (2.0, -3.0), (2.0, 1.0), (-4.0, 1.0)),
            ((-1.0, -4.0), (3.0, -4.0), (3.0, 2.0), (-1.0, 2.0)),
        )
        for turns, points in enumerate(expected):
            with self.subTest(turns=turns):
                rotated = outline.rotated_quarter_turns(turns)
                self.assertEqual(rotated.points, points)
                self.assertEqual(rotated.signed_area, 24.0)
                dimensions = (4.0, 6.0) if turns % 2 else (6.0, 4.0)
                self.assertEqual((rotated.width, rotated.height), dimensions)

    def test_quarter_turn_normalization_and_four_turn_identity(self):
        outline = CanonicalOutline.from_points(((1, 2), (5, 2), (2, 4)))
        for turns in (-9, -4, -3, -2, -1, 4, 5, 6, 7, 12):
            with self.subTest(turns=turns):
                self.assertEqual(outline.rotated_quarter_turns(turns),
                                 outline.rotated_quarter_turns(turns % 4))
        rotated = outline
        for _ in range(4):
            rotated = rotated.rotated_quarter_turns(1)
        self.assertEqual(rotated, outline)

    def test_transforms_return_new_immutable_canonical_instances(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        for transformed in (outline.translated(0, 0), outline.translated(2, 3),
                            *(outline.rotated_quarter_turns(i) for i in range(5))):
            with self.subTest(points=transformed.points):
                self.assertIsNot(transformed, outline)
                self.assertIsInstance(transformed.points, tuple)
                self.assertEqual(transformed.points[0], min(transformed.points))
                self.assertGreater(transformed.signed_area, 0)
                for point in transformed.points:
                    self.assertIsInstance(point, tuple)
                    self.assertTrue(all(type(value) is float for value in point))
                with self.assertRaises(TypeError):
                    transformed.points[0][0] = 99
                with self.assertRaises(FrozenInstanceError):
                    transformed.points = ()
        self.assertEqual(outline.points, RECTANGLE)

    def test_nonfinite_translation_is_rejected(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        for value in (float("nan"), float("inf"), -float("inf")):
            for offsets in ((value, 0), (0, value)):
                with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                    outline.translated(*offsets)

    def test_noninteger_quarter_turns_are_rejected(self):
        outline = CanonicalOutline.from_points(RECTANGLE)
        for turns in (1.5, 1.0, "1", None, True, False, float("nan"), float("inf")):
            with self.subTest(turns=turns), self.assertRaises(TypeError):
                outline.rotated_quarter_turns(turns)

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
