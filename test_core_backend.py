#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
