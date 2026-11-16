"""Re-export prefill-scenario pytest fixtures for test modules.

This conftest.py makes the @pytest.fixture objects from
``test_prefill_scenarios`` discoverable by pytest when tests import them.
"""

import sys
from pathlib import Path

_proxy_fixtures_dir = str(Path(__file__).resolve().parent)
if _proxy_fixtures_dir not in sys.path:
    sys.path.insert(0, _proxy_fixtures_dir)

# Import the fixture module so its @pytest.fixture objects are registered
# with pytest's fixture discovery mechanism.
import test_prefill_scenarios  # noqa: E402
