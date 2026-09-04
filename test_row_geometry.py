#!/usr/bin/env python3
import unittest

import numpy as np

from row_geometry import (
    PotPassCounter,
    RowFollowerConfig,
    detect_pot_station_xs,
    estimate_row_from_points,
)


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

    @staticmethod
    def _segmented_pot_side_points(side: str) -> np.ndarray:
        xs = [0.22 + 0.20 * idx for idx in range(6)]
        magnitudes = [0.30, 0.40, 0.31, 0.41, 0.30, 0.40]
        sign = 1.0 if side == "left" else -1.0
        points: list[tuple[float, float]] = []
        for x, magnitude in zip(xs, magnitudes):
            points.extend(((x - 0.025, sign * magnitude), (x + 0.025, sign * magnitude)))
        return np.repeat(np.asarray(points, dtype=np.float64), 2, axis=0)

    def test_front_segmented_left_pot_arcs_recover_fixed_width_centerline(self) -> None:
        self.cfg.segmented_pot_boundary_recovery = True

        estimate, _debug = estimate_row_from_points(
            self._segmented_pot_side_points("left"), self.cfg, 0.60
        )

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.mode, "left_only")
        self.assertEqual(estimate.boundary_source, "segmented_pot_left")
        self.assertTrue(estimate.segmented_boundary_recovered)
        self.assertAlmostEqual(estimate.center_line[1], 0.0, delta=0.03)

    def test_front_segmented_right_pot_arcs_recover_fixed_width_centerline(self) -> None:
        self.cfg.segmented_pot_boundary_recovery = True

        estimate, _debug = estimate_row_from_points(
            self._segmented_pot_side_points("right"), self.cfg, 0.60
        )

        self.assertTrue(estimate.found)
        self.assertEqual(estimate.mode, "right_only")
        self.assertEqual(estimate.boundary_source, "segmented_pot_right")
        self.assertTrue(estimate.segmented_boundary_recovered)
        self.assertAlmostEqual(estimate.center_line[1], 0.0, delta=0.03)

    def test_segmented_pot_recovery_is_disabled_by_default_for_reverse_rules(self) -> None:
        estimate, _debug = estimate_row_from_points(
            self._segmented_pot_side_points("left"), self.cfg, 0.60
        )

        self.assertFalse(estimate.found)
        self.assertFalse(estimate.segmented_boundary_recovered)


class PotCountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = RowFollowerConfig(
            forward_min=0.15,
            forward_max=1.60,
            lateral_limit=0.75,
            vehicle_half_width=0.20,
            safety_margin=0.04,
        )

    @staticmethod
    def _arc_points(station_xs: list[float], *, include_right: bool = True) -> np.ndarray:
        points: list[tuple[float, float]] = []
        # This follows LaserScan angular order: left is far-to-near and right
        # is near-to-far for objects in front of the vehicle.
        for x in reversed(station_xs):
            points.extend(((x + 0.04, 0.32), (x, 0.28), (x - 0.04, 0.32)))
        if include_right:
            for x in station_xs:
                points.extend(((x - 0.04, -0.32), (x, -0.28), (x + 0.04, -0.32)))
        return np.asarray(points, dtype=np.float64)

    def test_paired_pot_arcs_are_merged_into_longitudinal_stations(self) -> None:
        stations = detect_pot_station_xs(
            self._arc_points([0.35, 0.65, 0.95, 1.25]), self.cfg
        )

        self.assertEqual(len(stations), 4)
        np.testing.assert_allclose(stations, [0.35, 0.65, 0.95, 1.25], atol=0.02)

    def test_one_visible_side_still_produces_one_station_per_pot(self) -> None:
        stations = detect_pot_station_xs(
            self._arc_points([0.40, 0.75, 1.10], include_right=False), self.cfg
        )

        self.assertEqual(len(stations), 3)
        np.testing.assert_allclose(stations, [0.40, 0.75, 1.10], atol=0.02)

    def test_station_is_counted_once_when_it_crosses_vehicle_count_line(self) -> None:
        counter = PotPassCounter(count_line_x=0.45)

        self.assertEqual(counter.update([0.72], vehicle_progress_m=0.00), 0)
        self.assertEqual(counter.update([0.58], vehicle_progress_m=0.14), 0)
        self.assertEqual(counter.update([0.48], vehicle_progress_m=0.24), 0)
        self.assertEqual(counter.update([0.43], vehicle_progress_m=0.29), 1)
        self.assertEqual(counter.update([0.40], vehicle_progress_m=0.32), 0)
        self.assertEqual(counter.total_count, 1)

    def test_new_far_station_does_not_attach_to_already_passed_station(self) -> None:
        counter = PotPassCounter(count_line_x=0.45)
        for progress_m, station_x in ((0.00, 0.70), (0.18, 0.52), (0.27, 0.43)):
            counter.update([station_x], vehicle_progress_m=progress_m)

        self.assertEqual(counter.total_count, 1)
        self.assertEqual(counter.update([0.62], vehicle_progress_m=0.43), 0)
        self.assertEqual(counter.update([0.50], vehicle_progress_m=0.55), 0)
        self.assertEqual(counter.update([0.42], vehicle_progress_m=0.63), 1)
        self.assertEqual(counter.total_count, 2)

    def test_stationary_point_cloud_jitter_cannot_create_a_count(self) -> None:
        counter = PotPassCounter(count_line_x=0.45)

        for station_x in (0.72, 0.55, 0.43, 0.56, 0.41, 0.53, 0.42):
            counter.update([station_x], vehicle_progress_m=1.00)

        self.assertEqual(counter.total_count, 0)

    def test_shifted_duplicate_track_is_suppressed_by_station_spacing(self) -> None:
        counter = PotPassCounter(count_line_x=0.45, min_station_spacing_m=0.24)
        for progress_m, station_x in ((0.00, 0.72), (0.15, 0.57), (0.29, 0.43)):
            counter.update([station_x], vehicle_progress_m=progress_m)
        self.assertEqual(counter.total_count, 1)

        # The same arc is reconstructed 19 cm farther down the row, outside
        # association tolerance but still inside physical pot spacing.
        for progress_m, station_x in ((0.35, 0.56), (0.45, 0.46), (0.50, 0.41)):
            counter.update([station_x], vehicle_progress_m=progress_m)

        self.assertEqual(counter.total_count, 1)

    def test_reset_clears_row_count_before_row_change(self) -> None:
        counter = PotPassCounter(count_line_x=0.45)
        for progress_m, station_x in ((0.00, 0.70), (0.18, 0.52), (0.27, 0.43)):
            counter.update([station_x], vehicle_progress_m=progress_m)
        self.assertEqual(counter.total_count, 1)

        counter.reset()

        self.assertEqual(counter.total_count, 0)
        self.assertEqual(counter.tracks, [])
        self.assertEqual(counter.counted_station_positions, [])


if __name__ == "__main__":
    unittest.main()
