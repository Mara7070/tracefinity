"""Standalone single-outline layout tests using only the standard library."""

import unittest
from dataclasses import FrozenInstanceError

from app.services.object_geometry import CanonicalOutline, PlacedOutline
from app.services.object_layout import fit_candidates


class FitCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.rectangle = CanonicalOutline.from_points(((-2, -1), (4, -1), (4, 3), (-2, 3)))
        self.triangle = CanonicalOutline.from_points(((-2, -1), (4, -1), (-1, 3)))

    def test_rectangle_even_turns_fit_and_deduplicate(self):
        candidates = fit_candidates(self.rectangle, 7, 5)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (0,))
        self.assertEqual(candidates[0].bounds, (0, 0, 6, 4))
        # The rectangle's fitting 180-degree placement is identical to turn 0.
        rotated = PlacedOutline(self.rectangle, 4, 3, 2)
        self.assertEqual(rotated.points, candidates[0].points)

    def test_distinct_even_turns_only(self):
        candidates = fit_candidates(self.triangle, 7, 5)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (0, 2))

    def test_odd_turns_only(self):
        candidates = fit_candidates(self.triangle, 5, 7)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (1, 3))
        self.assertTrue(all(p.bounds == (0, 0, 4, 6) for p in candidates))

    def test_lower_left_alignment_and_max_bounds(self):
        candidates = fit_candidates(self.triangle, 10, 10, inset=0.5)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (0, 1, 2, 3))
        for candidate in candidates:
            with self.subTest(turn=candidate.quarter_turns):
                expected_max = (4.5, 6.5) if candidate.quarter_turns % 2 else (6.5, 4.5)
                self.assertEqual(candidate.bounds, (0.5, 0.5, *expected_max))

    def test_inset_reduces_space_to_no_fit(self):
        self.assertTrue(fit_candidates(self.rectangle, 6, 4))
        self.assertEqual(fit_candidates(self.rectangle, 6, 4, 0.25), ())

    def test_inset_consumes_region(self):
        for width, height, inset in ((6, 10, 3), (10, 6, 3), (6, 6, 4)):
            with self.subTest(width=width, height=height, inset=inset):
                self.assertEqual(fit_candidates(self.rectangle, width, height, inset), ())

    def test_exact_edge_fit_and_no_tolerance(self):
        candidates = fit_candidates(self.rectangle, 7, 5, 0.5)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bounds, (0.5, 0.5, 6.5, 4.5))
        self.assertEqual(fit_candidates(self.rectangle, 6 - 1e-12, 4), ())

    def test_square_deduplicates_all_rotations(self):
        square = CanonicalOutline.from_points(((2, 3), (4, 3), (4, 5), (2, 5)))
        candidates = fit_candidates(square, 4, 4, 1)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (0,))
        self.assertEqual(candidates[0].bounds, (1, 1, 3, 3))

    def test_equal_dimensions_retain_distinct_geometry_in_order(self):
        outline = CanonicalOutline.from_points(((0, 0), (4, 0), (1, 4)))
        candidates = fit_candidates(outline, 4, 4)
        self.assertEqual(tuple(p.quarter_turns for p in candidates), (0, 1, 2, 3))
        self.assertEqual(len({p.points for p in candidates}), 4)
        self.assertTrue(all((p.width, p.height) == (4, 4) for p in candidates))
        self.assertEqual(fit_candidates(outline, 4, 4), candidates)

    def test_preserves_outline_and_returns_immutable_placements(self):
        before = self.triangle.points
        candidates = fit_candidates(self.triangle, 10, 10)
        self.assertIsInstance(candidates, tuple)
        for candidate in candidates:
            self.assertIs(candidate.outline, self.triangle)
            with self.assertRaises(FrozenInstanceError):
                candidate.x = 99
        self.assertEqual(self.triangle.points, before)

    def test_invalid_dimensions_and_inset(self):
        for index in range(3):
            invalid = [float('nan'), float('inf'), -float('inf'), -1]
            if index < 2:
                invalid.append(0)
            for value in invalid:
                args = [10, 10, 0]
                args[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    fit_candidates(self.rectangle, *args)
        # Validation still applies when the inset would consume the region.
        with self.assertRaises(ValueError):
            fit_candidates(self.rectangle, 1, float('nan'), 10)


if __name__ == '__main__':
    unittest.main()
