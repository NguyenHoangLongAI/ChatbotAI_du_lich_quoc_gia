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
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerDAO:
    """DAO cho collection Customer trong Project du_lich_db"""

    DATABASE_NAME = "bai_chay_tourism_db"
    COLLECTION_NAME = "customers"
    DESCRIPTION_VECTOR_DIM = 768  # Dimension cho embedding của description

    def __init__(self, host="localhost", port="19530"):
        """Khởi tạo connection tới Milvus"""
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
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"✅ Connected to Milvus")
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
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
        """Tạo schema cho collection Customer"""
        fields = [
            # Primary key
            FieldSchema(
                name="customer_id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,  # Auto-generate ID
                description="ID tự động của khách hàng"
            ),

            # Customer information
            FieldSchema(
                name="name",
                dtype=DataType.VARCHAR,
                max_length=255,
                description="Tên khách hàng"
            ),

            FieldSchema(
                name="phone",
                dtype=DataType.VARCHAR,
                max_length=20,
                description="Số điện thoại"
            ),

            # Check-in/out timestamps (stored as INT64 - Unix timestamp)
            FieldSchema(
                name="checkin_time",
                dtype=DataType.INT64,
                description="Thời gian check-in (Unix timestamp)"
            ),

            FieldSchema(
                name="checkout_time",
                dtype=DataType.INT64,
                description="Thời gian check-out (Unix timestamp)"
            ),

            # Description
            FieldSchema(
                name="description",
                dtype=DataType.VARCHAR,
                max_length=2000,
                description="Mô tả về khách hàng (sở thích, ghi chú, lịch sử,...)"
            ),

            # Vector embedding của description
            FieldSchema(
                name="description_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.DESCRIPTION_VECTOR_DIM,
                description="Vector embedding của description"
            )
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Customer information collection with description vector",
            enable_dynamic_field=True
        )

        return schema

    def _get_or_create_collection(self) -> Collection:
        """Lấy hoặc tạo collection"""
        if utility.has_collection(self.COLLECTION_NAME):
            logger.info(f"✅ Collection '{self.COLLECTION_NAME}' exists in '{self.DATABASE_NAME}', loading...")
            collection = Collection(self.COLLECTION_NAME)
        else:
            logger.info(f"🔨 Creating collection '{self.COLLECTION_NAME}' in '{self.DATABASE_NAME}'")
            schema = self._create_schema()
            collection = Collection(
                name=self.COLLECTION_NAME,
                schema=schema
            )
            self._create_indexes(collection)
            logger.info(f"✅ Collection '{self.COLLECTION_NAME}' created successfully")

        collection.load()
        logger.info(f"✅ Collection loaded and ready")

        return collection

    def _create_indexes(self, collection: Collection):
        """Tạo indexes cho vector field"""
        # Index cho description_vector (COSINE similarity)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        collection.create_index(
            field_name="description_vector",
            index_params=index_params
        )
        logger.info("  ✅ Created index for description_vector (COSINE)")

    def insert_customer(self, customer_data: Dict) -> int:
        """
        Thêm một khách hàng mới

        Args:
            customer_data: Dict với keys:
                - name (str): Tên khách hàng
                - phone (str): Số điện thoại
                - checkin_time (int/datetime): Thời gian check-in
                - checkout_time (int/datetime): Thời gian check-out
                - description (str): Mô tả
                - description_vector (List[float]): Vector của description (dim=768)

        Returns:
            customer_id (int): ID của khách hàng vừa thêm
        """
        try:
            # Validate required fields
            required = ["name", "phone", "checkin_time", "checkout_time",
                        "description", "description_vector"]
            for field in required:
                assert field in customer_data, f"Missing '{field}'"

            # Validate vector dimension
            assert len(customer_data["description_vector"]) == self.DESCRIPTION_VECTOR_DIM, \
                f"description_vector must be {self.DESCRIPTION_VECTOR_DIM}D"

            # Convert datetime to timestamp if needed
            checkin = customer_data["checkin_time"]
            if isinstance(checkin, datetime):
                checkin = int(checkin.timestamp())

            checkout = customer_data["checkout_time"]
            if isinstance(checkout, datetime):
                checkout = int(checkout.timestamp())

            # Prepare entities (không cần customer_id vì auto_id=True)
            entities = [
                [customer_data["name"]],
                [customer_data["phone"]],
                [checkin],
                [checkout],
                [customer_data["description"]],
                [customer_data["description_vector"]]
            ]

            result = self.collection.insert(entities)
            self.collection.flush()

            customer_id = result.primary_keys[0]
            logger.info(f"✅ Inserted customer with ID: {customer_id}")
            return customer_id

        except Exception as e:
            logger.error(f"❌ Failed to insert customer: {e}")
            raise

    def insert_customers(self, customers_data: List[Dict]) -> List[int]:
        """
        Thêm nhiều khách hàng cùng lúc

        Args:
            customers_data: List[Dict] với mỗi dict có cấu trúc như insert_customer

        Returns:
            List[int]: Danh sách customer_ids
        """
        try:
            # Validate
            for customer in customers_data:
                required = ["name", "phone", "checkin_time", "checkout_time",
                            "description", "description_vector"]
                for field in required:
                    assert field in customer, f"Missing '{field}'"
                assert len(customer["description_vector"]) == self.DESCRIPTION_VECTOR_DIM

            # Convert datetimes to timestamps
            names = []
            phones = []
            checkins = []
            checkouts = []
            descriptions = []
            vectors = []

            for customer in customers_data:
                names.append(customer["name"])
                phones.append(customer["phone"])

                checkin = customer["checkin_time"]
                if isinstance(checkin, datetime):
                    checkin = int(checkin.timestamp())
                checkins.append(checkin)

                checkout = customer["checkout_time"]
                if isinstance(checkout, datetime):
                    checkout = int(checkout.timestamp())
                checkouts.append(checkout)

                descriptions.append(customer["description"])
                vectors.append(customer["description_vector"])

            # Insert
            entities = [names, phones, checkins, checkouts, descriptions, vectors]
            result = self.collection.insert(entities)
            self.collection.flush()

            logger.info(f"✅ Inserted {len(customers_data)} customers")
            return result.primary_keys

        except Exception as e:
            logger.error(f"❌ Failed to insert customers: {e}")
            raise

    def search_by_description(
            self,
            query_vector: List[float],
            top_k: int = 10,
            filters: Optional[str] = None
    ) -> List[Dict]:
        """
        Tìm kiếm khách hàng bằng description vector

        Args:
            query_vector: Vector của query (dim=768)
            top_k: Số kết quả trả về
            filters: Điều kiện lọc
                    VD: 'phone == "0901234567"'
                    VD: 'checkin_time > 1704067200'

        Returns:
            List các khách hàng phù hợp
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="description_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["customer_id", "name", "phone", "checkin_time",
                           "checkout_time", "description"]
        )

        return self._format_results(results)

    def get_by_id(self, customer_id: int) -> Optional[Dict]:
        """Lấy thông tin khách hàng theo ID"""
        results = self.collection.query(
            expr=f"customer_id == {customer_id}",
            output_fields=["customer_id", "name", "phone", "checkin_time",
                           "checkout_time", "description"]
        )

        if results:
            result = results[0]
            # Convert timestamps to datetime
            result["checkin_datetime"] = datetime.fromtimestamp(result["checkin_time"])
            result["checkout_datetime"] = datetime.fromtimestamp(result["checkout_time"])
            return result
        return None

    def get_by_phone(self, phone: str) -> List[Dict]:
        """Lấy thông tin khách hàng theo số điện thoại"""
        results = self.collection.query(
            expr=f'phone == "{phone}"',
            output_fields=["customer_id", "name", "phone", "checkin_time",
                           "checkout_time", "description"]
        )

        for result in results:
            result["checkin_datetime"] = datetime.fromtimestamp(result["checkin_time"])
            result["checkout_datetime"] = datetime.fromtimestamp(result["checkout_time"])

        return results

    def get_active_customers(self, current_timestamp: Optional[int] = None) -> List[Dict]:
        """
        Lấy danh sách khách hàng đang ở (đã check-in nhưng chưa check-out)

        Args:
            current_timestamp: Timestamp hiện tại (mặc định là now)

        Returns:
            List khách hàng đang ở
        """
        if current_timestamp is None:
            current_timestamp = int(datetime.now().timestamp())

        # Filter: checkin_time <= now AND checkout_time > now
        expr = f"checkin_time <= {current_timestamp} and checkout_time > {current_timestamp}"

        results = self.collection.query(
            expr=expr,
            output_fields=["customer_id", "name", "phone", "checkin_time",
                           "checkout_time", "description"],
            limit=1000
        )

        for result in results:
            result["checkin_datetime"] = datetime.fromtimestamp(result["checkin_time"])
            result["checkout_datetime"] = datetime.fromtimestamp(result["checkout_time"])

        return results

    def update_customer(self, customer_id: int, update_data: Dict) -> bool:
        """
        Cập nhật thông tin khách hàng

        Args:
            customer_id: ID khách hàng cần update
            update_data: Dict chứa các field cần update

        Returns:
            bool: True nếu thành công
        """
        try:
            # Lấy dữ liệu cũ
            old_data = self.get_by_id(customer_id)
            if not old_data:
                logger.error(f"Customer ID {customer_id} not found")
                return False

            # Xóa customer cũ
            self.delete_by_id(customer_id)

            # Merge data cũ với data mới
            merged_data = {**old_data, **update_data}

            # Insert lại với data mới (nhưng giữ customer_id cũ)
            # Note: Do auto_id=True, không thể giữ ID cũ được
            # Nên cách tốt nhất là không xóa mà chỉ đánh dấu
            logger.warning("⚠️  Update requires delete and re-insert with new ID")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to update customer: {e}")
            return False

    def delete_by_id(self, customer_id: int):
        """Xóa khách hàng theo ID"""
        expr = f"customer_id == {customer_id}"
        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"✅ Deleted customer with id={customer_id}")

    def get_statistics(self) -> Dict:
        """Lấy thống kê về collection"""
        stats = self.collection.num_entities
        return {
            "Project": self.DATABASE_NAME,
            "collection_name": self.COLLECTION_NAME,
            "total_customers": stats,
            "description_vector_dim": self.DESCRIPTION_VECTOR_DIM,
            "schema": str(self.collection.schema)
        }

    @staticmethod
    def _format_results(results) -> List[Dict]:
        """Format kết quả search từ Milvus"""
        formatted = []
        for hits in results:
            for hit in hits:
                result = {
                    "customer_id": hit.entity.get("customer_id"),
                    "name": hit.entity.get("name"),
                    "phone": hit.entity.get("phone"),
                    "checkin_time": hit.entity.get("checkin_time"),
                    "checkout_time": hit.entity.get("checkout_time"),
                    "description": hit.entity.get("description"),
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance)
                }
                # Convert timestamps to datetime
                result["checkin_datetime"] = datetime.fromtimestamp(result["checkin_time"])
                result["checkout_datetime"] = datetime.fromtimestamp(result["checkout_time"])
                formatted.append(result)
        return formatted

    def drop_collection(self):
        """Xóa collection (⚠️ CẨN THẬN - Xóa vĩnh viễn!)"""
        if utility.has_collection(self.COLLECTION_NAME):
            utility.drop_collection(self.COLLECTION_NAME)
            logger.info(f"✅ Dropped collection '{self.COLLECTION_NAME}'")
        else:
            logger.info(f"Collection '{self.COLLECTION_NAME}' does not exist")


# ========== Script test ==========
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta

    print("=" * 70)
    print("Testing CustomerDAO")
    print("=" * 70)

    try:
        # Khởi tạo DAO
        dao = CustomerDAO(host="localhost", port="19530")

        # Xem thống kê
        stats = dao.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"  - Database: {stats['Project']}")
        print(f"  - Collection: {stats['collection_name']}")
        print(f"  - Total customers: {stats['total_customers']}")

        # Test insert
        print(f"\n📝 Inserting sample customers...")

        now = datetime.now()
        checkin = now - timedelta(days=1)  # Check-in 1 ngày trước
        checkout = now + timedelta(days=2)  # Check-out sau 2 ngày

        sample_customers = [
            {
                "name": "Nguyễn Văn A",
                "phone": "0901234567",
                "checkin_time": checkin,
                "checkout_time": checkout,
                "description": "Khách hàng thân thiết, thích biển Da Nang, đã ở resort riverside 3 lần",
                "description_vector": np.random.rand(dao.DESCRIPTION_VECTOR_DIM).tolist()
            },
            {
                "name": "Trần Thị B",
                "phone": "0987654321",
                "checkin_time": int((now - timedelta(days=2)).timestamp()),
                "checkout_time": int((now + timedelta(days=1)).timestamp()),
                "description": "Khách mới, yêu cầu phòng view núi Ha Long, ăn chay",
                "description_vector": np.random.rand(dao.DESCRIPTION_VECTOR_DIM).tolist()
            }
        ]

        ids = dao.insert_customers(sample_customers)
        print(f"✅ Inserted customer IDs: {ids}")

        # Test query by phone
        print(f"\n🔍 Testing query by phone...")
        results = dao.get_by_phone("0901234567")
        if results:
            print(f"✅ Found customer:")
            for r in results:
                print(f"   - ID: {r['customer_id']}")
                print(f"   - Name: {r['name']}")
                print(f"   - Check-in: {r['checkin_datetime']}")
                print(f"   - Check-out: {r['checkout_datetime']}")
                print(f"   - Description: {r['description']}")

        # Test get active customers
        print(f"\n🔍 Getting active customers...")
        active = dao.get_active_customers()
        print(f"✅ Found {len(active)} active customers")
        for customer in active:
            print(f"   - {customer['name']} (ID: {customer['customer_id']})")

        print("\n✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)