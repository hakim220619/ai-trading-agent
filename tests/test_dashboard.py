from __future__ import annotations

import unittest

from app.api.routes import dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_uses_bootstrap_and_core_widgets(self) -> None:
        html = dashboard()
        self.assertIn("bootstrap@5.3.8", html)
        self.assertIn('id="balanceValue"', html)
        self.assertIn('id="profitCard"', html)
        self.assertIn("animateMetric", html)
        self.assertIn("profit-positive", html)
        self.assertIn('id="signalBadge"', html)
        self.assertIn('id="nearestSupport"', html)
        self.assertIn('id="nearestResistance"', html)
        self.assertIn('id="bosBadge"', html)
        self.assertIn('id="chochBadge"', html)
        self.assertIn('id="tradePlan"', html)
        self.assertIn('id="positionsBody"', html)
        self.assertIn("refreshDashboard()", html)


if __name__ == "__main__":
    unittest.main()
