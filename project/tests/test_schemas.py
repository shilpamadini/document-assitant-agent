import unittest
from datetime import datetime

from src.schemas import AnswerResponse, UserIntent


class TestSchemas(unittest.TestCase):
    def test_answer_response_creation(self):
        """AnswerResponse should accept required fields and auto-set timestamp."""
        ar = AnswerResponse(
            question="What is the total?",
            answer="The total is $10,000.",
            sources=["INV-001"],
            confidence=0.95,
        )
        self.assertEqual(ar.question, "What is the total?")
        self.assertEqual(ar.answer, "The total is $10,000.")
        self.assertEqual(ar.sources, ["INV-001"])
        self.assertTrue(0.0 <= ar.confidence <= 1.0)
        self.assertIsInstance(ar.timestamp, datetime)

    def test_user_intent_valid_types(self):
        """UserIntent should accept only allowed intent_type values."""
        intent = UserIntent(
            intent_type="qa",
            confidence=0.9,
            reasoning="The user asked a direct question.",
        )
        self.assertEqual(intent.intent_type, "qa")
        self.assertTrue(0.0 <= intent.confidence <= 1.0)

    def test_user_intent_invalid_type_raises(self):
        """Invalid intent_type should raise a validation error."""
        with self.assertRaises(ValueError):
            UserIntent(
                intent_type="invalid_intent",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="Invalid type for testing.",
            )


if __name__ == "__main__":
    unittest.main()
