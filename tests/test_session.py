"""Unit tests for conversational state and history management in Session."""

import unittest

from lclaude.session import Message, Session


class TestSessionMessage(unittest.TestCase):
    """Verifies dataclass conversion and equality for Message."""

    def test_to_dict_formatting(self) -> None:
        msg = Message(role="user", content="hello world")
        self.assertEqual(msg.to_dict(), {"role": "user", "content": "hello world"})

    def test_message_equality(self) -> None:
        msg1 = Message(role="assistant", content="test")
        msg2 = Message(role="assistant", content="test")
        self.assertEqual(msg1, msg2)


class TestSessionState(unittest.TestCase):
    """Verifies turn tracking, payload generation, and mutations in Session."""

    def test_initial_state_empty(self) -> None:
        session = Session()
        self.assertTrue(session.is_empty)
        self.assertEqual(session.turn_count, 0)
        self.assertEqual(session.messages, [])

    def test_system_prompt_injected_at_head(self) -> None:
        session = Session(system_prompt="You are a helpful coding assistant.")
        self.assertTrue(session.is_empty)
        self.assertEqual(
            session.messages,
            [{"role": "system", "content": "You are a helpful coding assistant."}],
        )

    def test_add_message_updates_state_and_turn_count(self) -> None:
        session = Session()
        session.add_message("user", "First prompt")

        self.assertFalse(session.is_empty)
        self.assertEqual(session.turn_count, 0)
        self.assertEqual(
            session.messages,
            [{"role": "user", "content": "First prompt"}],
        )

        session.add_message("assistant", "First response")
        self.assertEqual(session.turn_count, 1)
        self.assertEqual(len(session.messages), 2)

    def test_messages_returns_snapshot_copy(self) -> None:
        session = Session()
        session.add_message("user", "Hello")

        snapshot = session.messages
        snapshot.append({"role": "assistant", "content": "Mutated outside"})

        # Internal state must remain insulated from external modification
        self.assertEqual(len(session.messages), 1)

    def test_rollback_removes_unfulfilled_user_message(self) -> None:
        session = Session()
        session.add_message("user", "Pending question")

        rolled_back = session.rollback()

        self.assertTrue(rolled_back)
        self.assertTrue(session.is_empty)
        self.assertEqual(session.messages, [])

    def test_rollback_noop_when_assistant_is_last(self) -> None:
        session = Session()
        session.add_message("user", "Question")
        session.add_message("assistant", "Answer")

        rolled_back = session.rollback()

        self.assertFalse(rolled_back)
        self.assertEqual(len(session.messages), 2)

    def test_rollback_noop_on_empty_session(self) -> None:
        session = Session()
        self.assertFalse(session.rollback())

    def test_clear_resets_history_and_returns_turn_count(self) -> None:
        session = Session(system_prompt="Persistent system prompt")
        session.add_message("user", "Turn 1")
        session.add_message("assistant", "Ans 1")
        session.add_message("user", "Turn 2")
        session.add_message("assistant", "Ans 2")

        session.clear()
        self.assertTrue(session.is_empty)
        # System prompt survives clear
        self.assertEqual(
            session.messages,
            [{"role": "system", "content": "Persistent system prompt"}],
        )

    def test_get_preview_formatting_and_truncation(self) -> None:
        session = Session()
        session.add_message("user", "Line 1\nLine 2")
        session.add_message(
            "assistant",
            "This is a very long response that definitely exceeds twenty characters.",
        )

        previews = session.get_preview(max_chars=20)

        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[0], (1, "user", "Line 1 Line 2"))
        self.assertEqual(
            previews[1],
            (2, "assistant", "This is a very long ..."),
        )

    def test_get_preview_on_empty_session(self) -> None:
        session = Session()
        self.assertEqual(session.get_preview(), [])


if __name__ == "__main__":
    unittest.main()