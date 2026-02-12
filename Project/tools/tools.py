"""
tools.py — Các tool RAG kết nối với Milvus collections Bãi Cháy
"""

import json
import logging
from datetime import datetime
from typing import List

from langchain_core.tools import tool

from Project.crawl_baichay_service.tourism_dao import BaiChayTourismDAO
from Project.document_db.tourism_document_dao import TourismDocumentDAO
from Project.baichay_db.customer_dao import CustomerDAO
from Project.document_api.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAGTools:
    """Tập hợp các tool để tương tác với Milvus collections."""

    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530"):
        self.tourism_dao = BaiChayTourismDAO(host=milvus_host, port=milvus_port)
        self.document_dao = TourismDocumentDAO(host=milvus_host, port=milvus_port)
        self.customer_dao = CustomerDAO(host=milvus_host, port=milvus_port)
        self.embedding_service = EmbeddingService()
        logger.info("✅ RAG Tools initialized")

    # ------------------------------------------------------------------
    # Tourism search (direct call — không dùng @tool để giữ top_k linh hoạt)
    # ------------------------------------------------------------------

    def search_tourism_services(self, query: str, top_k: int = 5) -> str:
        """
        Tìm kiếm dịch vụ du lịch (tour, điểm đến, khách sạn, nhà hàng...).

        Returns:
            JSON string chứa danh sách dịch vụ kèm image_url và url bài viết.
        """
        try:
            query_vector = self.embedding_service.get_embedding(query)
            results = self.tourism_dao.search_by_description(
                query_vector=query_vector, top_k=top_k
            )

            formatted = []
            for r in results:
                formatted.append({
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "sub_type": r.get("sub_type"),
                    "location": r.get("location"),
                    "address": r.get("address"),
                    "description": r.get("description", "")[:500],
                    "price_range": r.get("price_range"),
                    "price_min": r.get("price_min"),
                    "price_max": r.get("price_max"),
                    "rating": r.get("rating"),
                    "opening_hours": r.get("opening_hours"),
                    "image_url": r.get("image_url", ""),   # ⭐ URL ảnh thumbnail
                    "url": r.get("url", ""),               # ⭐ URL bài viết
                    "similarity_score": round(r.get("score", 0), 3),
                })

            logger.info(f"✅ Found {len(formatted)} tourism services for: {query}")
            for i, item in enumerate(formatted, 1):
                logger.info(f"   [{i}] {item['name']}")
            return json.dumps(formatted, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ search_tourism_services error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Document search
    # ------------------------------------------------------------------

    @tool
    def search_documents(self, query: str, top_k: int = 3) -> str:
        """
        Tìm kiếm tài liệu quy định, hướng dẫn du lịch.

        Args:
            query: Câu hỏi về quy định hoặc thắc mắc.
            top_k: Số lượng chunks trả về.

        Returns:
            JSON string chứa nội dung tài liệu liên quan.
        """
        try:
            query_vector = self.embedding_service.get_embedding(query)
            search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

            results = self.document_dao.doc_collection.search(
                data=[query_vector],
                anns_field="description_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "description"],
            )

            formatted = []
            for hits in results:
                for hit in hits:
                    formatted.append({
                        "document_id": hit.entity.get("document_id"),
                        "content": hit.entity.get("description"),
                        "similarity_score": hit.score,
                    })
            return json.dumps(formatted, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ search_documents error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Get service by ID
    # ------------------------------------------------------------------

    @tool
    def get_service_by_id(self, service_id: int) -> str:
        """
        Lấy thông tin chi tiết dịch vụ theo ID.

        Args:
            service_id: ID của dịch vụ.

        Returns:
            JSON string với thông tin chi tiết.
        """
        try:
            result = self.tourism_dao.get_by_id(service_id)
            if result:
                return json.dumps(result, ensure_ascii=False, indent=2)
            return json.dumps({"error": "Service not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Create booking
    # ------------------------------------------------------------------

    @tool
    def create_customer_booking(
        self,
        name: str,
        phone: str,
        service_ids: List[int],
        service_descriptions: str,
        checkin_date: str,
        checkout_date: str,
    ) -> str:
        """
        Tạo booking cho khách hàng.

        Args:
            name: Họ tên khách hàng.
            phone: Số điện thoại.
            service_ids: Danh sách ID dịch vụ đã chọn.
            service_descriptions: Mô tả các dịch vụ.
            checkin_date: Ngày check-in (YYYY-MM-DD).
            checkout_date: Ngày check-out (YYYY-MM-DD).

        Returns:
            JSON string với thông tin booking đã tạo.
        """
        try:
            checkin = datetime.strptime(checkin_date, "%Y-%m-%d")
            checkout = datetime.strptime(checkout_date, "%Y-%m-%d")

            description = (
                f"Đặt dịch vụ du lịch Bãi Cháy. "
                f"Dịch vụ: {service_descriptions}. IDs: {service_ids}"
            )
            description_vector = self.embedding_service.get_embedding(description)

            customer_id = self.customer_dao.insert_customer({
                "name": name,
                "phone": phone,
                "checkin_time": checkin,
                "checkout_time": checkout,
                "description": description,
                "description_vector": description_vector,
            })

            result = {
                "status": "success",
                "customer_id": customer_id,
                "name": name,
                "phone": phone,
                "checkin": checkin_date,
                "checkout": checkout_date,
                "services": service_ids,
                "message": "Booking đã được tạo thành công!",
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ create_customer_booking error: {e}")
            return json.dumps({"status": "error", "message": str(e)})