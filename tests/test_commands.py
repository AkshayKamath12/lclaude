"""Unit tests for slash command handlers and dispatch dictionary."""

import io
import unittest
from unittest.mock import patch

from lclaude.commands import COMMANDS, handle_slash_command
from lclaude.session import Session


class TestCommandRegistryIntegrity(unittest.TestCase):
    """Verifies that the single source of truth dictionary is properly structured."""

    def test_commands_schema(self) -> None:
        """Every entry must start with '/', define a description, and have a callable handler."""
        self.assertGreater(len(COMMANDS), 0)
        for name, entry in COMMANDS.items():
            self.assertTrue(name.startswith("/"), f"Command '{name}' must start with '/'")
            self.assertIn("desc", entry)
            self.assertIsInstance(entry["desc"], str)
            self.assertIn("handler", entry)
            self.assertTrue(callable(entry["handler"]))


class TestSlashCommands(unittest.TestCase):
    """Verifies execution and terminal output of command handlers."""

    def setUp(self) -> None:
        self.session = Session()

    def test_clear_command_resets_session(self) -> None:
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi there")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/clear", self.session)

            self.assertTrue(handled)
            self.assertTrue(self.session.is_empty)
            self.assertIn("Cleared conversation history.", mock_stdout.getvalue())

    def test_history_command_empty_session(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/history", self.session)

            self.assertTrue(handled)
            self.assertIn("[History is currently empty.]", mock_stdout.getvalue())

    def test_history_command_populated_session(self) -> None:
        self.session.add_message("user", "What is 2+2?")
        self.session.add_message("assistant", "It is 4.")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/history", self.session)

            self.assertTrue(handled)
            output = mock_stdout.getvalue()
            self.assertIn("Active Context: 2 messages", output)
            self.assertIn("1. [user]: What is 2+2?", output)
            self.assertIn("2. [assistant]: It is 4.", output)

    def test_help_command_dynamically_lists_all_registered_commands(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/help", self.session)

            self.assertTrue(handled)
            output = mock_stdout.getvalue()
            self.assertIn("Available Commands:", output)
            # Confirms all commands from the dictionary appear in the output
            for cmd_name, info in COMMANDS.items():
                self.assertIn(cmd_name, output)
                self.assertIn(info["desc"], output)

    def test_exit_command_terminates_process(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with self.assertRaises(SystemExit) as ctx:
                handle_slash_command("/exit", self.session)

            self.assertEqual(ctx.exception.code, 0)
            self.assertIn("Exiting session.", mock_stdout.getvalue())

    def test_unknown_command_displays_guidance(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/invalid_command", self.session)

            self.assertTrue(handled)
            self.assertIn(
                "[Unknown command: '/invalid_command'. Type /help for options.]",
                mock_stdout.getvalue(),
            )

    def test_command_whitespace_and_casing(self) -> None:
        """Commands should tolerate leading/trailing whitespace and uppercase letters."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("   /HELP   ", self.session)

            self.assertTrue(handled)
            self.assertIn("Available Commands:", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()