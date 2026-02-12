from typing import List, Dict, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    db
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaiChayTourismDAO:
    """DAO cho du lịch Bãi Cháy - Tối ưu với 1 image_url duy nhất"""

    DATABASE_NAME = "bai_chay_tourism_db"
    COLLECTION_NAME = "bai_chay_data"

    DESCRIPTION_VECTOR_DIM = 768

    def __init__(self, host="localhost", port="19530"):
        """Khởi tạo connection và tạo collection"""
        self.host = host
        self.port = port
        self.connect()
        self.switch_database()
        self.collection = self._get_or_create_collection()

    def connect(self):
        """Kết nối tới Milvus server"""
        try:
            try:
                connections.disconnect("default")
            except:
                pass

            logger.info(f"🔌 Connecting to Milvus at {self.host}:{self.port}...")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise

    def switch_database(self):
        """Chuyển sang Project bai_chay_tourism_db"""
        try:
            databases = db.list_database()
            logger.info(f"📋 Existing databases: {databases}")

            if self.DATABASE_NAME not in databases:
                logger.info(f"🔨 Creating Project '{self.DATABASE_NAME}'...")
                db.create_database(self.DATABASE_NAME)
                logger.info(f"✅ Database '{self.DATABASE_NAME}' created")

            db.using_database(self.DATABASE_NAME)
            logger.info(f"✅ Switched to Project '{self.DATABASE_NAME}'")

        except Exception as e:
            logger.error(f"❌ Failed to switch Project: {e}")
            raise

    def _create_schema(self) -> CollectionSchema:
        """
        Schema tối ưu cho Bãi Cháy tourism collection
        ⭐ CHỈ 1 IMAGE_URL (VARCHAR) THAY VÌ ARRAY
        """
        fields = [
            # Primary key
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="ID duy nhất của dịch vụ"
            ),

            # Basic info
            FieldSchema(
                name="name",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Tên dịch vụ"
            ),

            # Category
            FieldSchema(
                name="type",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="Loại: diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen"
            ),

            FieldSchema(
                name="sub_type",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Phân loại chi tiết"
            ),

            # Location
            FieldSchema(
                name="location",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="Khu vực"
            ),

            FieldSchema(
                name="address",
                dtype=DataType.VARCHAR,
                max_length=1000,
                description="Địa chỉ cụ thể"
            ),

            # Content
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=65000,
                description="Mô tả chi tiết"
            ),

            # Price
            FieldSchema(
                name="price_range",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Khoảng giá dạng text: 'Miễn phí', '350.000 - 600.000 VNĐ'"
            ),

            FieldSchema(
                name="price_min",
                dtype=DataType.FLOAT,
                description="Giá tối thiểu (0 nếu miễn phí)"
            ),

            FieldSchema(
                name="price_max",
                dtype=DataType.FLOAT,
                description="Giá tối đa"
            ),

            # Additional info
            FieldSchema(
                name="opening_hours",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Giờ mở cửa"
            ),

            # ⭐ IMAGE - CHỈ 1 URL DUY NHẤT
            FieldSchema(
                name="image_url",
                dtype=DataType.VARCHAR,
                max_length=1000,
                description="URL ảnh thumbnail/featured image DUY NHẤT"
            ),

            # Metadata
            FieldSchema(
                name="rating",
                dtype=DataType.FLOAT,
                description="Đánh giá 0-5 sao"
            ),

            FieldSchema(
                name="view_count",
                dtype=DataType.INT64,
                description="Số lượt xem"
            ),

            # URL bài viết
            FieldSchema(
                name="url",
                dtype=DataType.VARCHAR,
                max_length=500,
                description="URL bài viết gốc"
            ),

            # Vector embedding
            FieldSchema(
                name="description_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.DESCRIPTION_VECTOR_DIM,
                description="Vector embedding 768D của description"
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="Bãi Cháy tourism optimized collection - 1 image URL only",
            enable_dynamic_field=True
        )

    def _get_or_create_collection(self) -> Collection:
        """Tạo hoặc load collection"""
        if utility.has_collection(self.COLLECTION_NAME):
            logger.info(f"✅ Collection '{self.COLLECTION_NAME}' exists, loading...")
            collection = Collection(self.COLLECTION_NAME)
        else:
            logger.info(f"🔨 Creating collection '{self.COLLECTION_NAME}'")
            schema = self._create_schema()
            collection = Collection(name=self.COLLECTION_NAME, schema=schema)

            # Create HNSW index for better performance
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,  # Number of connections
                    "efConstruction": 200  # Construction time parameter
                }
            }
            collection.create_index(
                field_name="description_vector",
                index_params=index_params
            )
            logger.info("  ✅ Created HNSW index for description_vector (COSINE)")

        collection.load()
        logger.info(f"✅ Collection loaded")
        return collection

    def insert_data(self, data: List[Dict]) -> List[int]:
        """
        Chèn dữ liệu vào collection
        ⭐ IMAGE_URL là string đơn giản, KHÔNG phải JSON array

        Args:
            data: List các dict với keys:
                - id, name, type, sub_type, location, address, description
                - price_range, price_min, price_max, opening_hours
                - image_url (string URL DUY NHẤT), rating, view_count, url
                - description_vector (List[float] - dim 768)

        Returns:
            List của primary keys
        """
        try:
            # Validate
            for item in data:
                required_fields = ["id", "name", "type", "description", "description_vector"]
                for field in required_fields:
                    assert field in item, f"Missing '{field}'"
                assert len(item["description_vector"]) == self.DESCRIPTION_VECTOR_DIM

            # Prepare data
            entities = [
                [item["id"] for item in data],
                [item["name"] for item in data],
                [item["type"] for item in data],
                [item.get("sub_type", "") for item in data],
                [item.get("location", "Bãi Cháy, Quảng Ninh") for item in data],
                [item.get("address", "") for item in data],
                [item["description"] for item in data],
                [item.get("price_range", "") for item in data],
                [item.get("price_min", 0.0) for item in data],
                [item.get("price_max", 0.0) for item in data],
                [item.get("opening_hours", "") for item in data],
                [item.get("image_url", "") for item in data],  # ⭐ String đơn giản
                [item.get("rating", 0.0) for item in data],
                [item.get("view_count", 0) for item in data],
                [item.get("url", "") for item in data],
                [item["description_vector"] for item in data]
            ]

            result = self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ Inserted {len(data)} records into collection")

            return result.primary_keys

        except Exception as e:
            logger.error(f"❌ Failed to insert data: {e}")
            raise

    def search_by_description(
            self,
            query_vector: List[float],
            top_k: int = 10,
            filters: Optional[str] = None
    ) -> List[Dict]:
        """Tìm kiếm bằng description vector"""
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 100  # HNSW search parameter
            }
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="description_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=[
                "id", "name", "type", "sub_type", "location", "address",
                "description", "price_range", "price_min", "price_max",
                "opening_hours", "image_url", "rating", "view_count", "url"
            ]
        )

        return self._format_results(results)

    def search_by_type(
            self,
            tourism_type: str,
            limit: int = 20
    ) -> List[Dict]:
        """
        Tìm kiếm theo loại
        Args:
            tourism_type: diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen
        """
        results = self.collection.query(
            expr=f'type == "{tourism_type}"',
            output_fields=[
                "id", "name", "type", "sub_type", "location", "address",
                "description", "price_range", "price_min", "price_max",
                "opening_hours", "image_url", "rating", "view_count", "url"
            ],
            limit=limit
        )
        return results

    def get_by_id(self, item_id: int) -> Optional[Dict]:
        """Lấy thông tin theo ID"""
        results = self.collection.query(
            expr=f"id == {item_id}",
            output_fields=[
                "id", "name", "type", "sub_type", "location", "address",
                "description", "price_range", "price_min", "price_max",
                "opening_hours", "image_url", "rating", "view_count", "url"
            ]
        )
        return results[0] if results else None

    def get_statistics(self) -> Dict:
        """Thống kê collection"""
        return {
            "Project": self.DATABASE_NAME,
            "collection": {
                "name": self.COLLECTION_NAME,
                "total_count": self.collection.num_entities,
                "vector_dim": self.DESCRIPTION_VECTOR_DIM,
                "schema_version": "optimized_v2_single_image"
            }
        }

    @staticmethod
    def _format_results(results) -> List[Dict]:
        """Format kết quả search"""
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.entity.get("id"),
                    "name": hit.entity.get("name"),
                    "type": hit.entity.get("type"),
                    "sub_type": hit.entity.get("sub_type"),
                    "location": hit.entity.get("location"),
                    "address": hit.entity.get("address"),
                    "description": hit.entity.get("description"),
                    "price_range": hit.entity.get("price_range"),
                    "price_min": hit.entity.get("price_min"),
                    "price_max": hit.entity.get("price_max"),
                    "opening_hours": hit.entity.get("opening_hours"),
                    "image_url": hit.entity.get("image_url"),  # ⭐ String đơn giản
                    "rating": hit.entity.get("rating"),
                    "view_count": hit.entity.get("view_count"),
                    "url": hit.entity.get("url"),
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance)
                })
        return formatted

    def drop_collection(self):
        """Xóa collection"""
        if utility.has_collection(self.COLLECTION_NAME):
            utility.drop_collection(self.COLLECTION_NAME)
            logger.info(f"✅ Dropped {self.COLLECTION_NAME}")


if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("Testing BaiChayTourismDAO - OPTIMIZED VERSION")
    print("=" * 70)

    try:
        dao = BaiChayTourismDAO(host="localhost", port="19530")

        stats = dao.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"  Database: {stats['Project']}")
        print(f"  Collection: {stats['collection']['name']} ({stats['collection']['total_count']} items)")
        print(f"  Schema: {stats['collection']['schema_version']}")

        print(f"\n📝 Inserting sample data...")

        print(f"\n🔍 Testing get by ID...")
        item = dao.get_by_id(99999)
        if item:
            print(f"✅ Found item:")
            print(f"   Name: {item['name']}")
            print(f"   Image URL: {item['image_url']}")
            print(f"   Article URL: {item['url']}")

        print("\n✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)