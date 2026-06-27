"""Tests for Cognee server lifecycle manager."""
import pytest
from unittest.mock import patch, MagicMock
import requests

import obsidianresumeforge.cognee_lifecycle as lifecycle
from obsidianresumeforge.cognee_lifecycle import ensure_cognee_running, cognee_available


def _reset_flag():
    lifecycle._cognee_available = False


def test_returns_true_when_health_200():
    _reset_flag()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = ensure_cognee_running("http://localhost:8000/health")
    assert result is True
    assert cognee_available() is True
    mock_get.assert_called_once()


def test_no_subprocess_when_already_reachable():
    _reset_flag()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("requests.get", return_value=mock_resp):
        with patch("subprocess.Popen") as mock_popen:
            ensure_cognee_running()
    mock_popen.assert_not_called()


def test_spawns_process_on_connection_error():
    _reset_flag()
    call_count = {"n": 0}

    def _get_side_effect(url, timeout):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.exceptions.ConnectionError("refused")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("requests.get", side_effect=_get_side_effect):
        with patch("subprocess.Popen") as mock_popen:
            with patch("time.sleep"):
                result = ensure_cognee_running()

    assert result is True
    mock_popen.assert_called_once()


def test_returns_false_when_persistently_unreachable():
    _reset_flag()
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with patch("subprocess.Popen"):
            with patch("time.sleep"):
                result = ensure_cognee_running()
    assert result is False
    assert cognee_available() is False


def test_returns_false_when_cognee_binary_missing():
    _reset_flag()
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with patch("subprocess.Popen", side_effect=FileNotFoundError("cognee not found")):
            result = ensure_cognee_running()
    assert result is False
    assert cognee_available() is False


def test_no_exception_raised_on_failure():
    _reset_flag()
    with patch("requests.get", side_effect=Exception("total chaos")):
        with patch("subprocess.Popen", side_effect=Exception("also chaos")):
            result = ensure_cognee_running()
    assert result is False
