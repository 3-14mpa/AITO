import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.messages import HumanMessage, AIMessage

# Import the functions to be tested from their correct location
from shared_components import search_memory_tool, search_knowledge_base_tool, list_uploaded_files_tool, read_full_document_tool


class TestContextAwareSearch(unittest.TestCase):
    """
    Tests the context-aware search functionality which reconstructs
    conversation threads.
    """

    @patch('shared_components.SQLChatMessageHistory')
    def test_search_reconstructs_full_conversation(self, mock_sql_history):
        # ARRANGE
        mock_vector_store = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.metadata = {'session_id': 'test-session-123'}
        mock_vector_store.similarity_search_with_score.return_value = [(mock_chunk, 0.1)]

        mock_history_instance = MagicMock()
        mock_history_instance.messages = [
            HumanMessage(content="Hello, how does this work?", name="Pimpa"),
            AIMessage(content="It reconstructs the full context!", name="ATOM1")
        ]
        mock_sql_history.return_value = mock_history_instance
        mock_config = {'user_id': 'Pimpa'}

        # ACT
        result = search_memory_tool(query="test query", config=mock_config, vector_store=mock_vector_store)

        # ASSERT
        self.assertIn("A keresés a 'test-session-123' azonosítójú beszélgetést találta a legrelevánsabbnak.", result)
        print("\n'test_search_reconstructs_full_conversation' ran successfully!")

    def test_search_memory_fallback_for_no_session_id(self):
        # ARRANGE
        mock_vector_store = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.page_content = "This is a contextless chunk."
        mock_chunk.metadata = {}
        mock_vector_store.similarity_search_with_score.return_value = [(mock_chunk, 0.2)]
        mock_config = {'user_id': 'Pimpa'}

        # ACT
        result = search_memory_tool(query="another query", config=mock_config, vector_store=mock_vector_store)

        # ASSERT
        self.assertIn("A memóriában talált releváns darabok (kontextus nélkül):", result)
        print("\n'test_search_memory_fallback_for_no_session_id' ran successfully!")


class TestKnowledgeBaseSearch(unittest.TestCase):
    """
    Tests the dedicated knowledge base search tool.
    """

    def test_search_knowledge_base_returns_formatted_string(self):
        # ARRANGE
        mock_docs_vector_store = MagicMock()
        mock_document_chunk = MagicMock()
        mock_document_chunk.page_content = "This is a snippet from a PDF about Flet."
        mock_document_chunk.metadata = {'source_document': 'flet_guide.pdf'}
        mock_docs_vector_store.similarity_search_with_score.return_value = [(mock_document_chunk, 0.2)]

        # ACT
        result = search_knowledge_base_tool(query="flet", config={}, docs_vector_store=mock_docs_vector_store)

        # ASSERT
        self.assertIn("Forrás: flet_guide.pdf", result)
        self.assertIn("Relevancia: 0.80", result)
        print("\n'test_search_knowledge_base_returns_formatted_string' ran successfully!")


class TestFileListingTool(unittest.TestCase):
    """
    Tests the tool for listing uploaded files from ChromaDB.
    """

    def test_list_files_returns_formatted_list(self):
        # ARRANGE
        mock_docs_vector_store = MagicMock()
        # Simulate the structure returned by Chroma's get() method
        mock_docs_vector_store.get.return_value = {
            'ids': ['1', '2', '3'],
            'metadatas': [
                {'source_document': 'file_a.pdf'},
                {'source_document': 'file_b.txt'},
                {'source_document': 'file_a.pdf'}
            ]
        }
        
        # ACT
        result = list_uploaded_files_tool(config={}, docs_vector_store=mock_docs_vector_store)

        # ASSERT
        self.assertIn("A Tudásbázisban a következő dokumentumok találhatók:", result)
        self.assertIn("- file_a.pdf", result)
        self.assertIn("- file_b.txt", result)
        self.assertEqual(result.count("file_a.pdf"), 1)
        print("\n'test_list_files_returns_formatted_list' ran successfully!")

    def test_list_files_handles_empty_knowledge_base(self):
        # ARRANGE
        mock_docs_vector_store = MagicMock()
        # Simulate an empty return from Chroma
        mock_docs_vector_store.get.return_value = {'ids': [], 'metadatas': []}

        # ACT
        result = list_uploaded_files_tool(config={}, docs_vector_store=mock_docs_vector_store)

        # ASSERT
        self.assertEqual(result, "A Tudásbázis jelenleg üres.")
        print("\n'test_list_files_handles_empty_knowledge_base' ran successfully!")


class TestReadFullDocumentTool(unittest.TestCase):
    """
    Tests the tool for reading a full document from ChromaDB.
    """

    def test_read_full_document_reconstructs_text(self):
        # ARRANGE
        mock_docs_vector_store = MagicMock()
        # Simulate the structure returned by Chroma's get() method
        mock_docs_vector_store.get.return_value = {
            'documents': ["world", "Hello, "],
            'metadatas': [
                {'source_document': 'test.txt', 'chunk_number': 2},
                {'source_document': 'test.txt', 'chunk_number': 1}
            ]
        }

        # ACT
        result = read_full_document_tool(filename="test.txt", docs_vector_store=mock_docs_vector_store)

        # ASSERT
        self.assertEqual("DOCUMENT_CONTENT:\nHello, world", result)
        print("\n'test_read_full_document_reconstructs_text' ran successfully!")

    def test_read_full_document_handles_not_found(self):
        # ARRANGE
        mock_docs_vector_store = MagicMock()
        # Simulate an empty return from Chroma
        mock_docs_vector_store.get.return_value = {'documents': [], 'metadatas': []}

        # ACT
        result = read_full_document_tool(filename="not_found.txt", docs_vector_store=mock_docs_vector_store)

        # ASSERT
        self.assertIn("Hiba: A 'not_found.txt' nevű dokumentum nem található a tudásbázisban.", result)
        print("\n'test_read_full_document_handles_not_found' ran successfully!")


if __name__ == '__main__':
    unittest.main()