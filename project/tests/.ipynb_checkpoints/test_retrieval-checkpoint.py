import unittest

from src.retrieval import SimulatedRetriever
from src.schemas import DocumentChunk


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.retriever = SimulatedRetriever()

    def test_retrieve_by_type_invoice(self):
        """Should retrieve at least one invoice document."""
        results = self.retriever.retrieve_by_type("invoice")
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], DocumentChunk)
        self.assertEqual(results[0].metadata.get("doc_type"), "invoice")

    def test_keyword_search(self):
        """Keyword search should return documents matching the query."""
        results = self.retriever.retrieve_by_keyword("insurance")
        self.assertGreater(len(results), 0)
        found_keyword = any(
            "insurance" in chunk.content.lower()
            or "insurance" in chunk.metadata.get("title", "").lower()
            for chunk in results
        )
        self.assertTrue(found_keyword)

    def test_statistics_structure(self):
        """Statistics should contain expected keys and types."""
        stats = self.retriever.get_statistics()
        self.assertIn("total_documents", stats)
        self.assertIn("documents_with_amounts", stats)
        self.assertIn("document_types", stats)
        self.assertIsInstance(stats["document_types"], dict)
        self.assertGreater(stats["total_documents"], 0)


if __name__ == "__main__":
    unittest.main()
