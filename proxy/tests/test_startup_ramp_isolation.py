"""Regression test for test-isolation leak: startup-ramp config must not leak
between tests (LP-0MUD0D0DU005R2VQ).

Within a single test file pytest executes methods in definition order, so this
ordering-explicit test is deterministic.
"""

import proxy.mode as mode_module


class TestStartupRampIsolation:
    """Order matters: test 01 simulates a test that enables the ramp,
    test 02 asserts the isolation fixture restored the global.
    """

    def test_01_simulates_enabled_ramp_leak(self):
        """Simulate a test that enables the startup ramp
        (like ``TestStartupRampConfig::test_default_config``).
        """
        mode_module.set_startup_ramp_config(None)
        # Verify: the global is now the enabled defaults
        assert mode_module._startup_ramp_config is not None
        assert mode_module._startup_ramp_config["enabled"] is True

    def test_02_isolation_fixture_restores_between_tests(self):
        """The autouse fixture must have restored
        ``_startup_ramp_config`` to its pre-test value (None) before
        this test started.
        """
        assert mode_module._startup_ramp_config is None
