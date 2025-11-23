import unittest

from src.prompts import (
    get_chat_prompt_template,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CALCULATION_SYSTEM_PROMPT,
)


class TestPrompts(unittest.TestCase):
    def test_qa_prompt_uses_qa_system_prompt(self):
        """QA intent should select QA_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("qa")
        rendered = template.invoke({"chat_history": [], "input": "Hello"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), QA_SYSTEM_PROMPT.strip())

    def test_summarization_prompt_uses_summarization_system_prompt(self):
        """Summarization intent should select SUMMARIZATION_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("summarization")
        rendered = template.invoke({"chat_history": [], "input": "Summarize this"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), SUMMARIZATION_SYSTEM_PROMPT.strip())

    def test_calculation_prompt_uses_calculation_system_prompt(self):
        """Calculation intent should select CALCULATION_SYSTEM_PROMPT."""
        template = get_chat_prompt_template("calculation")
        rendered = template.invoke({"chat_history": [], "input": "Compute totals"}).to_messages()
        system_msg = rendered[0]
        self.assertEqual(system_msg.content.strip(), CALCULATION_SYSTEM_PROMPT.strip())


if __name__ == "__main__":
    unittest.main()
