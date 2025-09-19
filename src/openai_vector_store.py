"""
OpenAI Vector Store Manager

Handles creation and management of OpenAI vector stores using the Files API
and vector store functionality for RAG with the Responses API.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreInfo:
    """Information about a created vector store"""
    vector_store_id: str
    name: str
    file_ids: List[str]
    file_count: int
    created_at: int
    status: str
    usage_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'vector_store_id': self.vector_store_id,
            'name': self.name,
            'file_ids': self.file_ids,
            'file_count': self.file_count,
            'created_at': self.created_at,
            'status': self.status,
            'usage_bytes': self.usage_bytes
        }


class OpenAIVectorStoreManager:
    """
    Manages OpenAI vector stores for RAG functionality.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize vector store manager.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        if not self.client.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

    def upload_files(self, file_paths: List[str],
                    purpose: str = "assistants") -> List[str]:
        """
        Upload files to OpenAI Files API.

        Args:
            file_paths: List of file paths to upload
            purpose: Purpose for the files (default: "assistants")

        Returns:
            List of file IDs from OpenAI
        """
        logger.info(f"Uploading {len(file_paths)} files to OpenAI...")

        file_ids = []
        upload_errors = []

        for i, file_path in enumerate(file_paths):
            try:
                # Progress logging
                if (i + 1) % 10 == 0 or i + 1 == len(file_paths):
                    logger.info(f"Upload progress: {i+1}/{len(file_paths)} files")

                with open(file_path, 'rb') as file:
                    response = self.client.files.create(
                        file=file,
                        purpose=purpose
                    )
                    file_ids.append(response.id)

            except Exception as e:
                error_msg = f"Failed to upload {file_path}: {str(e)}"
                logger.error(error_msg)
                upload_errors.append(error_msg)

                # Continue with other files
                continue

        logger.info(f"Upload completed: {len(file_ids)} successful, {len(upload_errors)} failed")

        if upload_errors:
            logger.warning(f"Upload errors encountered:")
            for error in upload_errors[:5]:  # Show first 5 errors
                logger.warning(f"  {error}")

        return file_ids

    def create_vector_store(self, name: str, file_ids: List[str]) -> VectorStoreInfo:
        """
        Create a vector store with uploaded files.

        Args:
            name: Name for the vector store
            file_ids: List of OpenAI file IDs to include

        Returns:
            VectorStoreInfo with store details
        """
        logger.info(f"Creating vector store '{name}' with {len(file_ids)} files...")

        try:
            # Create the vector store
            vector_store = self.client.beta.vector_stores.create(
                name=name,
                file_ids=file_ids
            )

            logger.info(f"Vector store created: {vector_store.id}")

            # Wait for processing to complete
            logger.info("Waiting for vector store processing to complete...")
            processed_store = self._wait_for_processing(vector_store.id)

            # Create info object
            store_info = VectorStoreInfo(
                vector_store_id=processed_store.id,
                name=processed_store.name,
                file_ids=file_ids,
                file_count=len(file_ids),
                created_at=processed_store.created_at,
                status=processed_store.status,
                usage_bytes=getattr(processed_store, 'usage_bytes', 0)
            )

            logger.info(f"Vector store '{name}' ready: {store_info.vector_store_id}")
            return store_info

        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise

    def _wait_for_processing(self, vector_store_id: str,
                           timeout: int = 1800) -> Any:  # 30 minute timeout
        """
        Wait for vector store processing to complete.

        Args:
            vector_store_id: ID of the vector store
            timeout: Maximum time to wait in seconds

        Returns:
            Processed vector store object
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                store = self.client.beta.vector_stores.retrieve(vector_store_id)

                if store.status == 'completed':
                    logger.info(f"Vector store processing completed")
                    return store
                elif store.status == 'failed':
                    raise Exception(f"Vector store processing failed")
                elif store.status in ['in_progress', 'pending']:
                    logger.info(f"Vector store status: {store.status}, waiting...")
                    time.sleep(30)  # Wait 30 seconds between checks
                else:
                    logger.warning(f"Unknown vector store status: {store.status}")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error checking vector store status: {str(e)}")
                time.sleep(30)

        raise TimeoutError(f"Vector store processing timed out after {timeout} seconds")

    def get_vector_store_info(self, vector_store_id: str) -> VectorStoreInfo:
        """
        Get information about an existing vector store.

        Args:
            vector_store_id: ID of the vector store

        Returns:
            VectorStoreInfo with current details
        """
        try:
            store = self.client.beta.vector_stores.retrieve(vector_store_id)

            # Get file IDs (may need to paginate)
            file_ids = []
            files = self.client.beta.vector_stores.files.list(vector_store_id)
            file_ids.extend([f.id for f in files.data])

            return VectorStoreInfo(
                vector_store_id=store.id,
                name=store.name,
                file_ids=file_ids,
                file_count=len(file_ids),
                created_at=store.created_at,
                status=store.status,
                usage_bytes=getattr(store, 'usage_bytes', 0)
            )

        except Exception as e:
            logger.error(f"Failed to get vector store info: {str(e)}")
            raise

    def list_vector_stores(self) -> List[VectorStoreInfo]:
        """
        List all vector stores in the account.

        Returns:
            List of VectorStoreInfo objects
        """
        try:
            stores = self.client.beta.vector_stores.list()

            store_infos = []
            for store in stores.data:
                # Get file count for each store
                files = self.client.beta.vector_stores.files.list(store.id)
                file_ids = [f.id for f in files.data]

                store_info = VectorStoreInfo(
                    vector_store_id=store.id,
                    name=store.name,
                    file_ids=file_ids,
                    file_count=len(file_ids),
                    created_at=store.created_at,
                    status=store.status,
                    usage_bytes=getattr(store, 'usage_bytes', 0)
                )
                store_infos.append(store_info)

            return store_infos

        except Exception as e:
            logger.error(f"Failed to list vector stores: {str(e)}")
            raise

    def delete_vector_store(self, vector_store_id: str) -> bool:
        """
        Delete a vector store.

        Args:
            vector_store_id: ID of the vector store to delete

        Returns:
            True if successful
        """
        try:
            self.client.beta.vector_stores.delete(vector_store_id)
            logger.info(f"Vector store {vector_store_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector store: {str(e)}")
            raise

    def create_vector_store_from_directory(self,
                                         directory_path: str,
                                         store_name: str,
                                         file_pattern: str = "*.txt") -> VectorStoreInfo:
        """
        Create a vector store from all files in a directory.

        Args:
            directory_path: Path to directory containing files
            store_name: Name for the vector store
            file_pattern: Pattern to match files (default: "*.txt")

        Returns:
            VectorStoreInfo for the created store
        """
        logger.info(f"Creating vector store from directory: {directory_path}")

        # Find all matching files
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        file_paths = list(directory.glob(file_pattern))
        if not file_paths:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {directory_path}")

        logger.info(f"Found {len(file_paths)} files to upload")

        # Upload files
        file_ids = self.upload_files([str(path) for path in file_paths])

        if not file_ids:
            raise Exception("No files were successfully uploaded")

        # Create vector store
        return self.create_vector_store(store_name, file_ids)

    def save_vector_store_info(self, store_info: VectorStoreInfo,
                              output_path: str):
        """
        Save vector store information to a JSON file.

        Args:
            store_info: VectorStoreInfo to save
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(store_info.to_dict(), f, indent=2)

            logger.info(f"Vector store info saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save vector store info: {str(e)}")
            raise