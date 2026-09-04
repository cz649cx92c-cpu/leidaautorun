#!/usr/bin/env python3
import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from plant_lidar_centerline_follower import (
    BodyCommand,
    CommandSender,
    ControlState,
    FORWARD_LOCAL_MIN_WZ_DEG,
    IOCommand,
    PlantRowFollower,
    RowEstimate,
    SteeringCommand,
    enforce_local_min_wz,
    limit_center_line_change,
)


class _FakeController:
    instances = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.body_commands = []
        self.steering_commands = []
        self.io_commands = []
        self.__class__.instances.append(self)

    def send_body(self, command) -> None:
        self.body_commands.append(command)

    def send_steering(self, command) -> None:
        self.steering_commands.append(command)

    def send_io(self, command) -> None:
        self.io_commands.append(command)

    def poll(self, limit=10):
        del limit
        return []

    def close(self) -> None:
        pass


class _RemoteReleaseController(_FakeController):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.poll_count = 0

    def poll(self, limit=10):
        del limit
        self.poll_count += 1
        remote_control = self.poll_count == 1
        return [
            {
                "name": "ctrl_fb",
                "data": {"gear": "park", "vx": 0.0, "vy": 0.0, "wz": 0.0},
            },
            {
                "name": "steering_ctrl_fb",
                "data": {"gear": "park", "wheel_speed_mps": 0.0, "wheel_angle_deg": 0.0},
            },
            {
                "name": "io_fb",
                "data": {
                    "unlock_ok": False,
                    "remote_control": remote_control,
                    "estop": False,
                },
            },
        ]


class CommandSenderCrabControlTests(unittest.TestCase):
    def _run_sender(self, body: BodyCommand) -> _FakeController:
        _FakeController.instances.clear()
        initial_state = ControlState(
            body=body,
            steering=SteeringCommand(gear="crab", speed=0.0, angle=0.0),
            io=IOCommand(),
        )
        sender = CommandSender(
            interface="test",
            channel="test",
            bitrate=500000,
            period_s=0.01,
            initial_state=initial_state,
        )

        with patch("plant_lidar_centerline_follower.FWMiniController", _FakeController):
            sender.start()
            time.sleep(0.06)
            sender.stop()

        return _FakeController.instances[0]

    def test_row_change_uses_body_vy_without_continuous_steering(self) -> None:
        controller = self._run_sender(
            BodyCommand(gear="crab", vx=0.0, vy=0.25, wz=0.0)
        )
        crab_commands = [
            command for command in controller.steering_commands if command.gear == "crab"
        ]
        self.assertEqual(len(crab_commands), 1)
        self.assertGreaterEqual(len(controller.body_commands), 3)
        self.assertTrue(any(abs(command.vy - 0.25) < 1e-9 for command in controller.body_commands))

    def test_non_lateral_crab_keeps_original_single_gear_sync(self) -> None:
        controller = self._run_sender(
            BodyCommand(gear="crab", vx=0.15, vy=0.0, wz=0.0)
        )
        crab_commands = [
            command for command in controller.steering_commands if command.gear == "crab"
        ]
        self.assertEqual(len(crab_commands), 1)


class CommandSenderRemoteReleaseTests(unittest.TestCase):
    def test_remote_release_reissues_drive_gear_handshake(self) -> None:
        _RemoteReleaseController.instances.clear()
        sender = CommandSender(
            interface="test",
            channel="test",
            bitrate=500000,
            period_s=0.01,
            initial_state=ControlState(
                body=BodyCommand(gear="4t4d", vx=0.07, vy=0.0, wz=0.0),
                steering=SteeringCommand(gear="4t4d", speed=0.0, angle=0.0),
                io=IOCommand(),
            ),
        )

        with patch(
            "plant_lidar_centerline_follower.FWMiniController",
            _RemoteReleaseController,
        ):
            sender.start()
            time.sleep(0.07)
            sender.stop()

        controller = _RemoteReleaseController.instances[0]
        gear_sync_commands = [
            command
            for command in controller.steering_commands
            if command.gear == "4t4d"
        ]
        self.assertGreaterEqual(len(gear_sync_commands), 2)


class LocalMinimumTurnTests(unittest.TestCase):
    def test_small_right_turn_is_boosted_to_negative_minimum(self) -> None:
        wz, boosted = enforce_local_min_wz(math.radians(-0.8), reverse=False)

        self.assertTrue(boosted)
        self.assertAlmostEqual(math.degrees(wz), -FORWARD_LOCAL_MIN_WZ_DEG)

    def test_small_left_turn_is_boosted_to_positive_minimum(self) -> None:
        wz, boosted = enforce_local_min_wz(math.radians(0.4), reverse=False)

        self.assertTrue(boosted)
        self.assertAlmostEqual(math.degrees(wz), FORWARD_LOCAL_MIN_WZ_DEG)

    def test_lost_forward_line_does_not_boost_stale_turn(self) -> None:
        wz, boosted = enforce_local_min_wz(
            math.radians(0.4),
            reverse=False,
            tracking_valid=False,
        )

        self.assertFalse(boosted)
        self.assertAlmostEqual(math.degrees(wz), 0.4)

    def test_zero_and_one_degree_commands_are_unchanged(self) -> None:
        zero_wz, zero_boosted = enforce_local_min_wz(0.0, reverse=False)
        one_wz, one_boosted = enforce_local_min_wz(math.radians(1.0), reverse=False)

        self.assertFalse(zero_boosted)
        self.assertAlmostEqual(zero_wz, 0.0)
        self.assertFalse(one_boosted)
        self.assertAlmostEqual(math.degrees(one_wz), 1.0)

    def test_reverse_small_turn_is_never_boosted(self) -> None:
        left_wz, left_boosted = enforce_local_min_wz(math.radians(0.5), reverse=True)
        right_wz, right_boosted = enforce_local_min_wz(math.radians(-0.5), reverse=True)

        self.assertFalse(left_boosted)
        self.assertFalse(right_boosted)
        self.assertAlmostEqual(math.degrees(left_wz), 0.5)
        self.assertAlmostEqual(math.degrees(right_wz), -0.5)


class ForwardCenterLineChangeTests(unittest.TestCase):
    def test_large_fresh_heading_change_is_limited_without_freezing_old_sign(self) -> None:
        reference_x = 0.6
        previous_heading_deg = 3.0
        current_heading_deg = -12.0
        previous_slope = math.tan(math.radians(previous_heading_deg))
        current_slope = math.tan(math.radians(current_heading_deg))
        previous = (previous_slope, 0.03 - previous_slope * reference_x)
        current = (current_slope, -0.05 - current_slope * reference_x)

        limited_line, limited = limit_center_line_change(
            previous,
            current,
            reference_x=reference_x,
            max_center_delta_m=0.06,
            max_heading_delta_deg=8.0,
        )

        self.assertTrue(limited)
        self.assertAlmostEqual(math.degrees(math.atan(limited_line[0])), -5.0)
        limited_y = limited_line[0] * reference_x + limited_line[1]
        self.assertAlmostEqual(limited_y, -0.03)


class ForwardLostLineTests(unittest.TestCase):
    def test_lost_line_keeps_forward_speed_but_clears_stale_turn(self) -> None:
        node = PlantRowFollower.__new__(PlantRowFollower)
        node.args = SimpleNamespace(gear="4t4d")
        node.last_good_time = time.monotonic()
        node.last_good_cmd = BodyCommand(
            gear="4t4d",
            vx=0.15,
            vy=0.0,
            wz=math.radians(0.8),
        )
        node.last_error_y = 0.03
        node.last_good_heading_deg = 3.0
        node.last_debug = {}
        node._set_debug_snapshot = lambda **values: node.last_debug.update(values)
        sent: list[BodyCommand] = []
        node._send_drive = lambda gear, vx, wz: sent.append(
            BodyCommand(gear=gear, vx=vx, vy=0.0, wz=wz)
        )

        held = node._hold_last_good_command(
            estimate=RowEstimate(found=False, mode="reject"),
            control_phase="lost_hold_forward",
            stop_reason="no_valid_boundary",
            warning="lost_hold",
            wz_limit_deg=0.6,
            wz_scale=0.0,
            max_age_s=None,
        )

        self.assertTrue(held)
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(sent[0].vx, 0.15)
        self.assertAlmostEqual(math.degrees(sent[0].wz), 0.0)


class ReverseSingleBoundaryControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = PlantRowFollower.__new__(PlantRowFollower)
        self.node.args = SimpleNamespace(
            gear="4t4d",
            reverse=True,
            max_wz_deg=5.0,
            reverse_max_wz_deg=5.0,
            max_heading_wz_deg=5.0,
            speed=0.15,
            min_speed=0.04,
            center_y_target=0.0,
            forward_center_left_offset_m=0.0,
            reverse_center_left_offset_m=0.0,
            forward_lookahead_x=0.6,
            control_deadband_y=0.004,
            reverse_steer_sign=-1.0,
            reverse_wz_filter_alpha=0.30,
            reverse_wz_enable_heading_deg=0.5,
            reverse_min_wz_error_y=0.025,
            reverse_min_wz_deg=1.8,
            reverse_heading_conflict_error_y=0.01,
            reverse_heading_max_ratio=0.35,
            reverse_sign_flip_guard_error_y=0.02,
            reverse_sign_flip_guard_last_wz_deg=1.5,
            max_wz_delta_deg_per_cycle=1.0,
            reverse_k_lat=24.0,
            reverse_k_heading=0.50,
            k_lat=1.2,
            k_heading=0.15,
            heading_conflict_error_y=0.01,
            heading_conflict_scale=0.0,
        )
        self.node.last_direct_error_y = 0.0
        self.node.last_error_y = 0.0
        self.node.last_cmd_wz = 0.0
        self.node.direct_error_history = []
        self.node.last_debug = {}
        self.node._set_debug_snapshot = lambda **values: self.node.last_debug.update(values)

    def _command_for_offset(self, offset: float, heading_deg: float = 0.0):
        slope = math.tan(math.radians(heading_deg))
        estimate = RowEstimate(
            found=True,
            mode="left_only",
            effective_mode="left_only",
            center_y=offset,
            raw_center_y=offset,
            heading_rad=math.radians(heading_deg),
            center_line=(slope, offset),
        )
        return self.node._make_command(estimate)

    def test_positive_offset_gets_negative_reverse_correction(self) -> None:
        command = self._command_for_offset(0.10)

        self.assertAlmostEqual(command.vx, -0.15)
        self.assertAlmostEqual(self.node.last_debug["target_wz_deg"], -2.4)
        self.assertAlmostEqual(math.degrees(command.wz), -1.0)
        self.assertEqual(self.node.last_debug["k_lat_eff"], 24.0)
        self.assertEqual(self.node.last_debug["max_wz_deg"], 5.0)

    def test_negative_offset_reverses_correction_direction(self) -> None:
        command = self._command_for_offset(-0.10)

        self.assertAlmostEqual(command.vx, -0.15)
        self.assertAlmostEqual(self.node.last_debug["target_wz_deg"], 2.4)
        self.assertAlmostEqual(math.degrees(command.wz), 1.0)

    def test_heading_error_is_corrected_when_lateral_error_is_near_zero(self) -> None:
        command = self._command_for_offset(0.06, heading_deg=-6.0)

        self.assertGreater(self.node.last_debug["target_wz_deg"], 2.5)
        self.assertAlmostEqual(math.degrees(command.wz), 1.0)

    def test_reverse_left_offset_tracks_centerline_five_cm_to_vehicle_right(self) -> None:
        self.node.args.reverse_center_left_offset_m = 0.05

        command = self._command_for_offset(-0.05)

        self.assertAlmostEqual(self.node.last_debug["center_y_target"], -0.05)
        self.assertAlmostEqual(self.node.last_debug["reverse_center_left_offset_m"], 0.05)
        self.assertAlmostEqual(self.node.last_debug["track_error_y"], 0.0)
        self.assertAlmostEqual(math.degrees(command.wz), 0.0)

    def test_forward_left_offset_tracks_centerline_four_cm_to_vehicle_right(self) -> None:
        self.node.args.reverse = False
        self.node.args.forward_center_left_offset_m = 0.04
        self.node.args.reverse_center_left_offset_m = 0.05

        self._command_for_offset(-0.04)

        self.assertAlmostEqual(self.node.last_debug["center_y_target"], -0.04)
        self.assertAlmostEqual(self.node.last_debug["forward_center_left_offset_m"], 0.04)
        self.assertAlmostEqual(self.node.last_debug["reverse_center_left_offset_m"], 0.0)
        self.assertAlmostEqual(self.node.last_debug["track_error_y"], 0.0)


if __name__ == "__main__":
    unittest.main()
