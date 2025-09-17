"""
Unit tests for main.py
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from spads import main
from spads.config import Config
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

class TestMain:
    """
    Test suite for the main function in main.py
    """

    @pytest.fixture(autouse=True)
    def setup(self, ):
        """
        Generic setup for the test suite.
        """

        # Mock the DictConfig
        self.mock_cfg = MagicMock(spec=DictConfig)

        # Mock the ConfigStore
        self.mock_cs = MagicMock(spec=ConfigStore)
        self.mock_cs.store(name="config", node=Config)


    def test_main_hydra_decorator(self):
        """
        Test the hydra decorator on the main is functioning.
        """

        assert main.main.__wrapped__.__name__ == "main", "main function \
                is not named 'main'"
        assert hasattr(main.main, "__wrapped__"), "main function \
                is not decorated with hydra.main"
        assert callable(main.main.__wrapped__), "main function \
                is not callable"
        assert main.main.__annotations__['cfg'] == 'DictConfig', "main function \
                does not take a DictConfig argument"


if __name__ == "__main__":
    pytest.main()  # Run the tests
        
