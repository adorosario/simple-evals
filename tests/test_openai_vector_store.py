"""
Unit tests for OpenAI Vector Store Manager
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.openai_vector_store import OpenAIVectorStoreManager, VectorStoreInfo


class TestVectorStoreInfo:
    """Test cases for VectorStoreInfo dataclass"""

    def test_vector_store_info_creation(self):
        """Test VectorStoreInfo creation and serialization"""
        store_info = VectorStoreInfo(
            vector_store_id="vs_123",
            name="Test Store",
            file_ids=["file_1", "file_2"],
            file_count=2,
            created_at=1234567890,
            status="completed",
            usage_bytes=1024
        )

        assert store_info.vector_store_id == "vs_123"
        assert store_info.name == "Test Store"
        assert store_info.file_count == 2
        assert len(store_info.file_ids) == 2

    def test_to_dict(self):
        """Test conversion to dictionary"""
        store_info = VectorStoreInfo(
            vector_store_id="vs_123",
            name="Test Store",
            file_ids=["file_1", "file_2"],
            file_count=2,
            created_at=1234567890,
            status="completed"
        )

        result = store_info.to_dict()

        assert result['vector_store_id'] == "vs_123"
        assert result['name'] == "Test Store"
        assert result['file_ids'] == ["file_1", "file_2"]
        assert result['file_count'] == 2
        assert result['status'] == "completed"


class TestOpenAIVectorStoreManager:
    """Test cases for OpenAIVectorStoreManager"""

    def setup_method(self):
        """Set up test fixtures"""
        # Mock OpenAI client to avoid real API calls
        with patch('src.openai_vector_store.OpenAI') as mock_openai:
            self.mock_client = MagicMock()
            mock_openai.return_value = self.mock_client
            self.mock_client.api_key = "test_key"

            self.manager = OpenAIVectorStoreManager(api_key="test_key")

    def test_init_with_api_key(self):
        """Test initialization with API key"""
        with patch('src.openai_vector_store.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.api_key = "test_key"

            manager = OpenAIVectorStoreManager(api_key="test_key")
            assert manager.client == mock_client

    def test_init_without_api_key(self):
        """Test initialization without API key raises error"""
        with patch('src.openai_vector_store.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.api_key = None

            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIVectorStoreManager()

    def test_upload_files_success(self):
        """Test successful file upload"""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            file1_path = os.path.join(temp_dir, "test1.txt")
            file2_path = os.path.join(temp_dir, "test2.txt")

            with open(file1_path, 'w') as f:
                f.write("Test content 1")
            with open(file2_path, 'w') as f:
                f.write("Test content 2")

            # Mock file upload responses
            mock_response1 = MagicMock()
            mock_response1.id = "file_123"
            mock_response2 = MagicMock()
            mock_response2.id = "file_456"

            self.mock_client.files.create.side_effect = [mock_response1, mock_response2]

            # Test upload
            file_ids = self.manager.upload_files([file1_path, file2_path])

            assert len(file_ids) == 2
            assert file_ids == ["file_123", "file_456"]
            assert self.mock_client.files.create.call_count == 2

    def test_upload_files_with_errors(self):
        """Test file upload with some failures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1_path = os.path.join(temp_dir, "test1.txt")
            file2_path = os.path.join(temp_dir, "test2.txt")

            with open(file1_path, 'w') as f:
                f.write("Test content 1")
            with open(file2_path, 'w') as f:
                f.write("Test content 2")

            # Mock responses - first succeeds, second fails
            mock_response = MagicMock()
            mock_response.id = "file_123"

            self.mock_client.files.create.side_effect = [
                mock_response,
                Exception("Upload failed")
            ]

            file_ids = self.manager.upload_files([file1_path, file2_path])

            assert len(file_ids) == 1
            assert file_ids == ["file_123"]

    def test_create_vector_store_success(self):
        """Test successful vector store creation"""
        # Mock vector store creation
        mock_store = MagicMock()
        mock_store.id = "vs_123"
        mock_store.name = "Test Store"
        mock_store.created_at = 1234567890
        mock_store.status = "completed"

        self.mock_client.beta.vector_stores.create.return_value = mock_store
        self.mock_client.beta.vector_stores.retrieve.return_value = mock_store

        file_ids = ["file_1", "file_2"]
        store_info = self.manager.create_vector_store("Test Store", file_ids)

        assert isinstance(store_info, VectorStoreInfo)
        assert store_info.vector_store_id == "vs_123"
        assert store_info.name == "Test Store"
        assert store_info.file_ids == file_ids
        assert store_info.file_count == 2
        assert store_info.status == "completed"

    @patch('src.openai_vector_store.time.sleep')  # Speed up test
    def test_wait_for_processing_completed(self, mock_sleep):
        """Test waiting for vector store processing completion"""
        # Mock store that starts in_progress then becomes completed
        mock_store_processing = MagicMock()
        mock_store_processing.status = "in_progress"

        mock_store_completed = MagicMock()
        mock_store_completed.status = "completed"

        self.mock_client.beta.vector_stores.retrieve.side_effect = [
            mock_store_processing,
            mock_store_completed
        ]

        result = self.manager._wait_for_processing("vs_123")
        assert result.status == "completed"

    @patch('src.openai_vector_store.time.sleep')
    def test_wait_for_processing_failed(self, mock_sleep):
        """Test waiting for vector store processing that fails"""
        mock_store = MagicMock()
        mock_store.status = "failed"

        self.mock_client.beta.vector_stores.retrieve.return_value = mock_store

        with pytest.raises(Exception, match="Vector store processing failed"):
            self.manager._wait_for_processing("vs_123")

    @patch('src.openai_vector_store.time.time')
    @patch('src.openai_vector_store.time.sleep')
    def test_wait_for_processing_timeout(self, mock_sleep, mock_time):
        """Test waiting for vector store processing timeout"""
        # Mock time to simulate timeout
        mock_time.side_effect = [0, 1801]  # Start at 0, then past timeout

        mock_store = MagicMock()
        mock_store.status = "in_progress"

        self.mock_client.beta.vector_stores.retrieve.return_value = mock_store

        with pytest.raises(TimeoutError, match="processing timed out"):
            self.manager._wait_for_processing("vs_123", timeout=1800)

    def test_get_vector_store_info(self):
        """Test getting vector store information"""
        # Mock store retrieval
        mock_store = MagicMock()
        mock_store.id = "vs_123"
        mock_store.name = "Test Store"
        mock_store.created_at = 1234567890
        mock_store.status = "completed"

        # Mock files list
        mock_files = MagicMock()
        mock_file1 = MagicMock()
        mock_file1.id = "file_1"
        mock_file2 = MagicMock()
        mock_file2.id = "file_2"
        mock_files.data = [mock_file1, mock_file2]

        self.mock_client.beta.vector_stores.retrieve.return_value = mock_store
        self.mock_client.beta.vector_stores.files.list.return_value = mock_files

        store_info = self.manager.get_vector_store_info("vs_123")

        assert store_info.vector_store_id == "vs_123"
        assert store_info.name == "Test Store"
        assert store_info.file_ids == ["file_1", "file_2"]
        assert store_info.file_count == 2

    def test_list_vector_stores(self):
        """Test listing all vector stores"""
        # Mock stores list
        mock_store1 = MagicMock()
        mock_store1.id = "vs_123"
        mock_store1.name = "Store 1"
        mock_store1.created_at = 1234567890
        mock_store1.status = "completed"

        mock_store2 = MagicMock()
        mock_store2.id = "vs_456"
        mock_store2.name = "Store 2"
        mock_store2.created_at = 1234567891
        mock_store2.status = "completed"

        mock_stores = MagicMock()
        mock_stores.data = [mock_store1, mock_store2]

        # Mock files for each store
        mock_files = MagicMock()
        mock_files.data = []

        self.mock_client.beta.vector_stores.list.return_value = mock_stores
        self.mock_client.beta.vector_stores.files.list.return_value = mock_files

        stores = self.manager.list_vector_stores()

        assert len(stores) == 2
        assert stores[0].vector_store_id == "vs_123"
        assert stores[1].vector_store_id == "vs_456"

    def test_delete_vector_store(self):
        """Test deleting a vector store"""
        self.mock_client.beta.vector_stores.delete.return_value = None

        result = self.manager.delete_vector_store("vs_123")

        assert result is True
        self.mock_client.beta.vector_stores.delete.assert_called_once_with("vs_123")

    def test_create_vector_store_from_directory(self):
        """Test creating vector store from directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1_path = os.path.join(temp_dir, "doc1.txt")
            file2_path = os.path.join(temp_dir, "doc2.txt")

            with open(file1_path, 'w') as f:
                f.write("Document 1 content")
            with open(file2_path, 'w') as f:
                f.write("Document 2 content")

            # Mock upload and vector store creation
            mock_upload_response1 = MagicMock()
            mock_upload_response1.id = "file_123"
            mock_upload_response2 = MagicMock()
            mock_upload_response2.id = "file_456"

            self.mock_client.files.create.side_effect = [
                mock_upload_response1,
                mock_upload_response2
            ]

            mock_store = MagicMock()
            mock_store.id = "vs_123"
            mock_store.name = "Test Store"
            mock_store.created_at = 1234567890
            mock_store.status = "completed"

            self.mock_client.beta.vector_stores.create.return_value = mock_store
            self.mock_client.beta.vector_stores.retrieve.return_value = mock_store

            # Test creation
            store_info = self.manager.create_vector_store_from_directory(
                temp_dir, "Test Store"
            )

            assert store_info.vector_store_id == "vs_123"
            assert store_info.name == "Test Store"
            assert store_info.file_count == 2

    def test_create_vector_store_from_directory_no_files(self):
        """Test creating vector store from directory with no matching files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-txt file
            file_path = os.path.join(temp_dir, "doc.pdf")
            with open(file_path, 'w') as f:
                f.write("PDF content")

            with pytest.raises(ValueError, match="No files found matching pattern"):
                self.manager.create_vector_store_from_directory(
                    temp_dir, "Test Store"
                )

    def test_save_vector_store_info(self):
        """Test saving vector store info to JSON"""
        store_info = VectorStoreInfo(
            vector_store_id="vs_123",
            name="Test Store",
            file_ids=["file_1", "file_2"],
            file_count=2,
            created_at=1234567890,
            status="completed"
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            self.manager.save_vector_store_info(store_info, output_path)

            # Verify file was created and has correct content
            assert os.path.exists(output_path)

            with open(output_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data['vector_store_id'] == "vs_123"
            assert saved_data['name'] == "Test Store"
            assert saved_data['file_count'] == 2

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)