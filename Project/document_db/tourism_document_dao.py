# Project/tourism_document_dao.py
"""
Milvus Manager cho Tourism Document Collection:
- document_tour: Document embeddings cho tài liệu tư vấn du lịch
- document_tour_urls: Document URLs cho tourism documents

Sử dụng Project: du_lich_bai_chay
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    db
)
from typing import List, Dict, Any
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TourismDocumentDAO:
    """Milvus manager cho tourism document collections trong Project du_lich_bai_chay"""

    def __init__(
            self,
            host: str = "localhost",
            port: str = "19530",
            embedding_dim: int = 768,
            database_name: str = "bai_chay_tourism_db"
    ):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.database_name = database_name

        # Collection names
        self.doc_collection_name = "document_tour"
        self.url_collection_name = "document_tour_urls"

        # Collections
        self.doc_collection = None
        self.url_collection = None

        # State
        self.is_initialized = False

        # Field limits
        self.max_id_length = 190
        self.max_document_id_length = 90
        self.max_description_length = 60000

        # Embedding model (lazy loading)
        self._embedding_model = None

    # ==================== CONNECTION ====================

    async def initialize(self, max_retries: int = 5, retry_delay: int = 2):
        """Initialize Milvus connection and create tourism document collections"""
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting to connect to Milvus at {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Auto-detect Docker environment
                if self.host == "milvus":
                    import socket
                    try:
                        socket.gethostbyname("milvus")
                    except socket.gaierror:
                        self.host = "localhost"
                        logger.warning("⚠️ Running outside Docker, using localhost")

                # Disconnect existing connection
                try:
                    connections.disconnect("default")
                except:
                    pass

                # Connect
                connections.connect("default", host=self.host, port=self.port)
                logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")

                # Switch to tourism Project
                try:
                    databases = db.list_database()
                    if self.database_name not in databases:
                        logger.info(f"📁 Database '{self.database_name}' not found. Creating...")
                        db.create_database(self.database_name)
                        logger.info(f"✅ Database '{self.database_name}' created")

                    db.using_database(self.database_name)
                    logger.info(f"✅ Using Project: {self.database_name}")

                except Exception as e:
                    logger.error(f"❌ Failed to setup Project '{self.database_name}': {e}")
                    raise

                # Create collections
                await self.create_document_collection()
                await self.create_url_collection()

                self.is_initialized = True
                logger.info("✅ Tourism Document DAO initialization completed")
                return True

            except Exception as e:
                logger.error(
                    f"❌ Milvus initialization error "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed to initialize after {max_retries} attempts")
                    self.is_initialized = False
                    raise e

    def _check_initialized(self):
        """Check if Milvus is initialized"""
        if not self.is_initialized:
            raise Exception(
                "Milvus is not initialized. Please check connection."
            )

    # ==================== EMBEDDING MODEL ====================

    @property
    def embedding_model(self):
        """Lazy load embedding model (FORCE CPU)"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("🔄 Loading Vietnamese SBERT model (CPU mode)...")
                device = 'cpu'

                self._embedding_model = SentenceTransformer(
                    'keepitreal/vietnamese-sbert',
                    device=device
                )

                logger.info(f"✅ Embedding model loaded on {device}")

            except ImportError:
                logger.error("❌ sentence-transformers not installed")
                raise ImportError("Run: pip install sentence-transformers")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise

        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim

            embedding = self.embedding_model.encode(
                text.strip(),
                normalize_embeddings=True
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            return [0.0] * self.embedding_dim

    # ==================== COLLECTION CREATION ====================

    async def create_document_collection(self):
        """Create document_tour collection"""
        try:
            if utility.has_collection(self.doc_collection_name):
                logger.info(f"📦 Collection {self.doc_collection_name} already exists")
                self.doc_collection = Collection(self.doc_collection_name)
                self.doc_collection.load()
                logger.info(f"✅ Loaded existing collection {self.doc_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.doc_collection_name}")

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    is_primary=True
                ),
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=100
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=65000
                ),
                FieldSchema(
                    name="description_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Tourism document embeddings for travel regulations and advice (768D)"
            )

            self.doc_collection = Collection(
                name=self.doc_collection_name,
                schema=schema,
                using='default'
            )

            # HNSW index for fast search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }

            self.doc_collection.create_index(
                field_name="description_vector",
                index_params=index_params
            )

            self.doc_collection.load()
            logger.info(f"✅ Collection {self.doc_collection_name} created with HNSW index")

        except Exception as e:
            logger.error(f"❌ Document collection creation error: {e}")
            raise e

    async def create_url_collection(self):
        """Create document_tour_urls collection"""
        try:
            if utility.has_collection(self.url_collection_name):
                logger.info(f"📦 Collection {self.url_collection_name} already exists")
                self.url_collection = Collection(self.url_collection_name)
                self.url_collection.load()
                logger.info(f"✅ Loaded existing collection {self.url_collection_name}")
                return

            logger.info(f"🔨 Creating NEW collection: {self.url_collection_name}")

            fields = [
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True,
                    description="Unique document identifier"
                ),
                FieldSchema(
                    name="url",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Public URL to the document"
                ),
                FieldSchema(
                    name="filename",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    description="Original filename"
                ),
                FieldSchema(
                    name="file_type",
                    dtype=DataType.VARCHAR,
                    max_length=20,
                    description="File extension"
                ),
                FieldSchema(
                    name="filename_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                    description="Filename embedding for semantic search"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Tourism document URLs with filename embeddings"
            )

            self.url_collection = Collection(
                name=self.url_collection_name,
                schema=schema,
                using='default'
            )

            # Index for filename search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            self.url_collection.create_index(
                field_name="filename_vector",
                index_params=index_params
            )

            self.url_collection.load()
            logger.info(f"✅ Collection {self.url_collection_name} created successfully")

        except Exception as e:
            logger.error(f"❌ URL collection creation error: {e}")
            raise e

    # ==================== DOCUMENT EMBEDDINGS ====================

    async def insert_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Insert document embeddings with progress logging"""
        try:
            self._check_initialized()

            if not self.doc_collection:
                raise Exception("Document collection not initialized")

            try:
                self.doc_collection.load()
            except:
                pass

            if not embeddings_data:
                return 0

            field_limits = {
                "id": self.max_id_length,
                "document_id": self.max_document_id_length,
                "description": self.max_description_length
            }

            validated_data = []
            for item in embeddings_data:
                if not all(key in item for key in
                           ["id", "document_id", "description", "description_vector"]):
                    continue

                validated_item = self._validate_and_truncate(item, field_limits)

                if len(validated_item["description_vector"]) != self.embedding_dim:
                    continue

                validated_data.append(validated_item)

            if not validated_data:
                return 0

            ids = [item["id"] for item in validated_data]
            document_ids = [item["document_id"] for item in validated_data]
            descriptions = [item["description"] for item in validated_data]
            vectors = [item["description_vector"] for item in validated_data]

            batch_size = 100
            total_inserted = 0
            total_batches = (len(validated_data) + batch_size - 1) // batch_size

            for i in range(0, len(validated_data), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_document_ids = document_ids[i:i + batch_size]
                batch_descriptions = descriptions[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]

                entities = [
                    batch_ids,
                    batch_document_ids,
                    batch_descriptions,
                    batch_vectors
                ]

                try:
                    self.doc_collection.insert(entities)
                    total_inserted += len(batch_ids)

                    current_batch = (i // batch_size) + 1
                    if current_batch % 10 == 0 or current_batch == total_batches:
                        logger.info(
                            f"Inserted batch {current_batch}/{total_batches}: "
                            f"{total_inserted} items"
                        )

                except Exception as batch_error:
                    logger.error(f"Error inserting batch {i // batch_size + 1}: {batch_error}")
                    continue

            self.doc_collection.flush()
            logger.info(f"✅ Total inserted to TOURISM: {total_inserted} embeddings")
            return total_inserted

        except Exception as e:
            logger.error(f"❌ Insert error: {e}")
            raise e

    async def delete_document(self, document_id: str) -> bool:
        """Delete all embeddings for a document"""
        try:
            self._check_initialized()

            if not self.doc_collection:
                raise Exception("Document collection not initialized")

            expr = f'document_id == "{document_id}"'
            self.doc_collection.delete(expr)

            logger.info(f"✅ Deleted from TOURISM: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Document delete error: {e}")
            return False

    # ==================== DOCUMENT URLS ====================

    def insert_url(
            self,
            document_id: str,
            url: str,
            filename: str = "",
            file_type: str = ""
    ) -> bool:
        """Insert or update document URL with filename embedding"""
        try:
            if not self.url_collection:
                raise Exception("URL collection not initialized")

            # Validate and truncate
            if len(document_id) > 100:
                document_id = document_id[:100]
            if len(url) > 500:
                logger.warning(f"URL too long, truncating: {url[:50]}...")
                url = url[:500]
            if len(filename) > 200:
                filename = filename[:200]
            if len(file_type) > 20:
                file_type = file_type[:20]

            # Generate filename embedding
            logger.info(f"🔄 Embedding filename: {filename}")
            filename_embedding = self.embed_text(filename)

            # Delete existing entry if any
            try:
                expr = f'document_id == "{document_id}"'
                existing = self.url_collection.query(
                    expr=expr,
                    output_fields=["document_id"],
                    limit=1
                )

                if existing:
                    self.url_collection.delete(expr=f'document_id in ["{document_id}"]')
                    logger.debug(f"Deleted existing entry for {document_id}")
            except Exception as del_error:
                logger.debug(f"No existing entry or delete failed: {del_error}")

            # Insert new entry
            entities = [
                [document_id],
                [url],
                [filename],
                [file_type],
                [filename_embedding]
            ]

            self.url_collection.insert(entities)
            self.url_collection.flush()

            logger.info(f"✅ Inserted URL to TOURISM: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inserting URL: {e}")
            return False

    def get_url(self, document_id: str) -> dict:
        """Get URL for a document"""
        try:
            expr = f'document_id == "{document_id}"'

            results = self.url_collection.query(
                expr=expr,
                output_fields=["url", "filename", "file_type"],
                limit=1
            )

            if results:
                return {
                    "document_id": document_id,
                    "url": results[0].get("url"),
                    "filename": results[0].get("filename"),
                    "file_type": results[0].get("file_type")
                }

            return None

        except Exception as e:
            logger.error(f"❌ Error getting URL: {e}")
            return None

    def delete_url(self, document_id: str) -> bool:
        """Delete URL entry"""
        try:
            expr = f'document_id in ["{document_id}"]'
            self.url_collection.delete(expr)
            self.url_collection.flush()
            logger.info(f"✅ Deleted URL from TOURISM: {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False

    def search_by_filename(
            self,
            query: str,
            top_k: int = 5,
            min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search documents by filename using semantic search"""
        try:
            query_vector = self.embed_text(query)

            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }

            results = self.url_collection.search(
                data=[query_vector],
                anns_field="filename_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "url", "filename", "file_type"]
            )

            documents = []
            for hits in results:
                for hit in hits:
                    if hit.score >= min_score:
                        documents.append({
                            "document_id": hit.entity.get("document_id"),
                            "url": hit.entity.get("url"),
                            "filename": hit.entity.get("filename"),
                            "file_type": hit.entity.get("file_type"),
                            "similarity_score": hit.score
                        })

            logger.info(f"✅ Found {len(documents)} documents for query: {query}")
            return documents

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    # ==================== UTILITIES ====================

    def _validate_and_truncate(
            self,
            data: Dict[str, Any],
            field_limits: Dict[str, int]
    ) -> Dict[str, Any]:
        """Validate and truncate fields"""
        validated = data.copy()

        for field, max_length in field_limits.items():
            if field in validated and isinstance(validated[field], str):
                if len(validated[field]) > max_length:
                    validated[field] = validated[field][:max_length - 3] + "..."

        return validated

    async def health_check(self) -> bool:
        """Check Milvus connection health"""
        try:
            if not self.is_initialized:
                return False
            connections.get_connection_addr("default")
            return True
        except:
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for collections"""
        try:
            stats = {"initialized": self.is_initialized}

            if self.doc_collection:
                self.doc_collection.load()
                stats["document_tour"] = {
                    "count": self.doc_collection.num_entities,
                    "name": self.doc_collection_name
                }
                indexes = self.doc_collection.indexes
                if indexes:
                    stats["document_tour"]["index_type"] = indexes[0].params.get(
                        'index_type', 'unknown'
                    )

            if self.url_collection:
                self.url_collection.load()
                stats["document_tour_urls"] = {
                    "count": self.url_collection.num_entities,
                    "name": self.url_collection_name
                }

            return stats

        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {"error": str(e)}


async def main():
    """Test the Tourism Document DAO"""

    dao = TourismDocumentDAO(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        embedding_dim=768,
        database_name="bai_chay_tourism_db"
    )

    try:
        await dao.initialize()
        logger.info("🎉 Tourism Document collections are READY")

        # Health check
        if await dao.health_check():
            logger.info("💚 Milvus health check PASSED")
        else:
            logger.warning("⚠️ Milvus health check FAILED")

        # Print stats
        stats = await dao.get_collection_stats()
        print("\n=== Collection Statistics ===")
        import json
        print(json.dumps(stats, indent=2))

    except Exception as e:
        logger.error(f"🔥 Failed to setup tourism document DAO: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())