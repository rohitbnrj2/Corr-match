"""
Tests for the experiment module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spads.experiment import ExperimentConfigs, run_experiment


class TestExperiment:
    """
    Test cases for experiment module.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Setup for all tests.
        """

        self.mock_cfg = MagicMock()
        self.mock_cfg.exp = MagicMock(spec=ExperimentConfigs)
        self.mock_cfg.exp.seed = 123
        self.mock_cfg.exp.is_cuda = False
        self.mock_cfg.exp.algorithm = "sift"

    def test_exp_reproducibility(self):
        """
        Test setting the reproducibility.
        """

        try:
            run_experiment(self.mock_cfg)

        except Exception as e:
            pytest.fail(f"run_experiment raised an exception: {e}")

        assert self.mock_cfg.exp.seed == 123

    def test_experiment_runner(self):
        """
        Test running the experiment with default configs. Two algorithms
        testing is sufficient to ensure code if functioning.
        """

        with patch(
            "spads.experiment.sift_runner", return_value=0.95
        ) as mock_sift_runner:
            result = run_experiment(self.mock_cfg)

            assert result == 0.95
            mock_sift_runner.assert_called_once()

        self.mock_cfg.exp.algorithm = "sim"
        with patch("spads.experiment.sim_runner", return_value=None) as mock_sim_runner:
            result = run_experiment(self.mock_cfg)

            assert result == 0.85
            mock_sim_runner.assert_called_once()

    def test_invalid_algorithm(self):
        """
        Test handling of invalid algorithm.
        """

        self.mock_cfg.exp.algorithm = "invalid_algo"

        with pytest.raises(
            ValueError,
            match=f"{self.mock_cfg.exp.algorithm.capitalize()} module not found",
        ):
            run_experiment(self.mock_cfg)

    def test_cuda_requirement(self):
        """
        Test handling of CUDA / PyTorch requirement.
        """

        self.mock_cfg.exp.is_cuda = True

        with patch("spads.experiment.torch", None):
            with pytest.raises(
                RuntimeError,
                match="PyTorch is required for CUDA experiments but not installed",
            ):
                run_experiment(self.mock_cfg)

            with patch("spads.experiment.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False

                with pytest.raises(
                    RuntimeError, match="CUDA unavailable, require CUDA for experiment"
                ):
                    run_experiment(self.mock_cfg)


if __name__ == "__main__":
    pytest.main()
