#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from backend import build_parser
from core_backend import _can_interface_is_up


class CanInterfaceReadinessTests(unittest.TestCase):
    def _interface(self, *, operstate: str, flags: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        interface = Path(temp_dir.name)
        (interface / "operstate").write_text(operstate, encoding="utf-8")
        (interface / "flags").write_text(flags, encoding="utf-8")
        return interface

    def test_administratively_up_socketcan_is_ready_when_operstate_is_down(self) -> None:
        interface = self._interface(operstate="down\n", flags="0x40081\n")

        self.assertTrue(_can_interface_is_up(interface))

    def test_down_interface_without_iff_up_is_not_ready(self) -> None:
        interface = self._interface(operstate="down\n", flags="0x40080\n")

        self.assertFalse(_can_interface_is_up(interface))


class PotStopFlagTests(unittest.TestCase):
    @staticmethod
    def _parse(*extra: str):
        return build_parser().parse_args(
            ["autorun", "--db", "map.bin", "--mission", "mission.json", *extra]
        )

    def test_pot_stop_is_disabled_by_default(self) -> None:
        self.assertFalse(self._parse().lidar_pot_stop_enabled)

    def test_pot_stop_can_be_enabled_explicitly(self) -> None:
        self.assertTrue(
            self._parse("--lidar-pot-stop-enabled").lidar_pot_stop_enabled
        )


if __name__ == "__main__":
    unittest.main()
