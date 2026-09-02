#!/usr/bin/env python3
import unittest

import numpy as np

from row_geometry import RowFollowerConfig, estimate_row_from_points


class PairedMidpointCenterlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = RowFollowerConfig(
            forward_min=0.15,
            forward_max=1.60,
            lateral_limit=0.75,
            min_points=16,
            min_line_bins=4,
            min_side_points_per_bin=2,
            min_row_width=0.48,
            max_row_width=0.78,
            bin_size=0.20,
            vehicle_half_width=0.20,
            safety_margin=0.03,
        )

    @staticmethod
    def _pot_arc_points(
        widths: list[float],
        centers: list[float] | None = None,
    ) -> np.ndarray:
        xs = [0.22 + 0.20 * idx for idx in range(len(widths))]
        centers = centers or [0.02] * len(widths)
        points: list[tuple[float, float]] = []
        for x, width, center in zip(xs, widths, centers):
            for dx in (-0.025, 0.025):
                points.append((x + dx, center + 0.5 * width))
                points.append((x + dx, center - 0.5 * width))
        return np.asarray(points, dtype=np.float64)

    def test_discrete_pot_arcs_recover_center_from_paired_bins(self) -> None:
        points = self._pot_arc_points([0.62, 0.76, 0.50, 0.74, 0.56, 0.70])

        estimate, _debug = estimate_row_from_points(points, self.cfg, 0.60)

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.mode, "both_sides")
        self.assertEqual(estimate.boundary_source, "paired_midpoints")
        self.assertGreaterEqual(estimate.paired_bins, 4)
        self.assertAlmostEqual(estimate.center_y, 0.02, delta=0.02)
        self.assertFalse(estimate.left_valid)
        self.assertFalse(estimate.right_valid)

    def test_one_bad_midpoint_is_rejected_without_losing_corridor(self) -> None:
        centers = [0.02, 0.02, 0.02, 0.24, 0.02, 0.02]
        points = self._pot_arc_points([0.62] * len(centers), centers)

        estimate, _debug = estimate_row_from_points(points, self.cfg, 0.60)

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.boundary_source, "paired_midpoints")
        self.assertEqual(estimate.paired_bins, 5)
        self.assertAlmostEqual(estimate.center_y, 0.02, delta=0.02)

    def test_out_of_range_corridor_width_does_not_activate_midpoints(self) -> None:
        points = self._pot_arc_points([0.82, 0.96, 0.84, 0.94, 0.86, 0.92])

        estimate, _debug = estimate_row_from_points(points, self.cfg, 0.60)

        self.assertNotEqual(estimate.boundary_source, "paired_midpoints")

    def test_fewer_than_four_paired_bins_does_not_activate_midpoints(self) -> None:
        points = self._pot_arc_points([0.62, 0.76, 0.50])
        points = np.repeat(points, 2, axis=0)

        estimate, _debug = estimate_row_from_points(points, self.cfg, 0.60)

        self.assertNotEqual(estimate.boundary_source, "paired_midpoints")

    @staticmethod
    def _single_boundary_points(y: float) -> np.ndarray:
        points: list[tuple[float, float]] = []
        for idx in range(6):
            x = 0.22 + 0.20 * idx
            points.extend(((x - 0.025, y), (x + 0.025, y)))
        return np.repeat(np.asarray(points, dtype=np.float64), 2, axis=0)

    def test_left_boundary_uses_fixed_sixty_centimeter_channel(self) -> None:
        estimate, _debug = estimate_row_from_points(
            self._single_boundary_points(0.36), self.cfg, 0.60
        )

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.mode, "left_only")
        self.assertAlmostEqual(estimate.center_line[1], 0.06, delta=0.01)
        self.assertAlmostEqual(estimate.right_line[1], -0.24, delta=0.01)

    def test_right_boundary_uses_fixed_sixty_centimeter_channel(self) -> None:
        estimate, _debug = estimate_row_from_points(
            self._single_boundary_points(-0.34), self.cfg, 0.60
        )

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.mode, "right_only")
        self.assertAlmostEqual(estimate.center_line[1], -0.04, delta=0.01)
        self.assertAlmostEqual(estimate.left_line[1], 0.26, delta=0.01)


if __name__ == "__main__":
    unittest.main()
