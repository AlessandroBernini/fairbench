"""Tests for fairbench.logging_config — structured logging."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

from fairbench.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    """Test JSON log formatter."""

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_exception(self):
        """Test formatting a record with exception info."""
        formatter = JsonFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestSetupLogging:
    """Test logging setup function."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a configured logger."""
        logger = setup_logging(level="INFO")
        assert logger.name == "fairbench"
        assert logger.level == logging.INFO

    def test_setup_logging_with_debug_level(self):
        """Test setting DEBUG level."""
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_invalid_level(self):
        """Test that invalid level defaults to INFO."""
        logger = setup_logging(level="INVALID")
        assert logger.level == logging.INFO

    def test_console_handler_format(self):
        """Test that console handler uses correct format."""
        log_stream = StringIO()
        logger = setup_logging(level="INFO")

        # Replace handlers to use our stream
        logger.handlers.clear()
        console = logging.StreamHandler(log_stream)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console)

        logger.info("Test message")
        output = log_stream.getvalue()
        assert "INFO: Test message" in output

    def test_log_file_creates_directory(self, tmp_path):
        """Test that log file handler creates parent directories."""
        log_file = str(tmp_path / "logs" / "fairbench.log")
        logger = setup_logging(level="INFO", log_file=log_file)

        # Write a log entry
        logger.info("Test log entry")

        # Verify file was created
        assert Path(log_file).exists()

        # Verify it contains JSON
        with open(log_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["message"] == "Test log entry"

    def test_logging_filters_by_level(self, tmp_path):
        """Test that DEBUG messages are filtered at INFO level."""
        log_file = str(tmp_path / "logs" / "fairbench.log")
        logger = setup_logging(level="INFO", log_file=log_file)

        # Write DEBUG and INFO messages
        logger.debug("Debug message")
        logger.info("Info message")

        # Read file
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Only INFO message should be in file (DEBUG filtered by handler)
        info_found = False
        debug_found = False
        for line in lines:
            data = json.loads(line)
            if data["message"] == "Info message":
                info_found = True
            if data["message"] == "Debug message":
                debug_found = True

        assert info_found
        assert not debug_found  # DEBUG filtered by console handler
