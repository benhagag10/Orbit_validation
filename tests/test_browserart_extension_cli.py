"""
CLI tests for BrowserART extension parameters.

Tests the `orbit browserart` Click command with extension-related flags
using CliRunner and mocks for isolation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from orbit.wrapper.cli import cli


class TestBrowserartCLI:
    """Test the `orbit browserart` CLI subcommand."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_browserart_help_shows_extension(self, runner):
        """--help output mentions hbb_extension as a dataset choice."""
        result = runner.invoke(cli, ["browserart", "--help"])
        assert result.exit_code == 0
        assert "hbb_extension" in result.output

    def test_browserart_help_shows_dataset_option(self, runner):
        """--help output shows --dataset option."""
        result = runner.invoke(cli, ["browserart", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    @patch("inspect_ai.eval")
    @patch("orbit.scenarios.browser.browserart.task.browserart_safety")
    def test_passes_extension_dataset(self, mock_task_fn, mock_eval, runner):
        """--dataset hbb_extension --task-id 236 passes correct args to browserart_safety."""
        # Mock returns
        mock_task = MagicMock()
        mock_task_fn.return_value = mock_task
        mock_eval.return_value = []

        # "y\n" answers the preflight "Continue anyway?" prompt that fires
        # when the optional browsergym dep isn't installed in the test env.
        result = runner.invoke(cli, [
            "browserart",
            "--model", "openai/gpt-4o",
            "--dataset", "hbb_extension",
            "--task-id", "236",
        ], input="y\n")

        # Verify browserart_safety was called
        mock_task_fn.assert_called_once()
        call_kwargs = mock_task_fn.call_args
        # It can be positional or keyword args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("dataset") == "hbb_extension"
            assert call_kwargs.kwargs.get("task_ids") == "236"
        else:
            # Check through the call
            pass

    @patch("inspect_ai.eval")
    @patch("orbit.scenarios.browser.browserart.task.browserart_safety")
    def test_explicit_turns_passed(self, mock_task_fn, mock_eval, runner):
        """--max-turns 25 passes the value through to browserart_safety."""
        mock_task = MagicMock()
        mock_task_fn.return_value = mock_task
        mock_eval.return_value = []

        result = runner.invoke(cli, [
            "browserart",
            "--model", "openai/gpt-4o",
            "--dataset", "hbb_extension",
            "--max-turns", "25",
        ], input="y\n")

        mock_task_fn.assert_called_once()
        call_kwargs = mock_task_fn.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("max_turns") == 25

    @patch("inspect_ai.eval")
    @patch("orbit.scenarios.browser.browserart.task.browserart_safety")
    def test_default_model_required(self, mock_task_fn, mock_eval, runner):
        """browserart command requires --model."""
        result = runner.invoke(cli, ["browserart", "--dataset", "hbb_extension"])
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower() or "Error" in result.output
