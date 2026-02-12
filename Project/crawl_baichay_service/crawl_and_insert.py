#!/usr/bin/env python3
"""
Script tích hợp: Crawl dữ liệu từ website và insert vào Milvus
⭐ OPTIMIZED VERSION - CHỈ 1 ẢNH VÀ URL BÀI VIẾT
"""
import sys
import json
import logging
from typing import Dict

# Import crawler và DAO
sys.path.append('/mnt/user-data/uploads')
from crawler_baichay import BaiChayCrawler
from tourism_dao import BaiChayTourismDAO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrawlAndInsertPipeline:
    """Pipeline để crawl và insert dữ liệu vào Milvus - OPTIMIZED"""

    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530"):
        """
        Khởi tạo pipeline
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        logger.info("🚀 Initializing OPTIMIZED Crawl & Insert Pipeline...")

        # Khởi tạo crawler
        logger.info("📡 Initializing crawler...")
        self.crawler = BaiChayCrawler()

        # Khởi tạo DAO
        logger.info("💾 Connecting to Milvus...")
        self.dao = BaiChayTourismDAO(host=milvus_host, port=milvus_port)

        logger.info("✅ Pipeline initialized successfully!")
        logger.info("⭐ Using OPTIMIZED schema: 1 image_url only")

    def crawl_category_and_insert(
            self,
            category_type: str,
            max_items: int = None,
            max_pages: int = 10,
            batch_size: int = 10
    ) -> Dict:
        """
        Crawl một category và insert vào Milvus

        Args:
            category_type: Loại category (diem-den, luu-tru, tour, etc.)
            max_items: Số items tối đa để crawl
            max_pages: Số trang tối đa để crawl
            batch_size: Số items trong mỗi batch insert

        Returns:
            Dict với thống kê: crawled_count, inserted_count, failed_count
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🎯 Processing category: {category_type}")
        logger.info(f"{'=' * 80}")

        # Crawl data
        logger.info(f"📡 Crawling {category_type}...")
        crawled_data = self.crawler.crawl_category(
            category_type=category_type,
            max_items=max_items,
            max_pages=max_pages
        )

        if not crawled_data:
            logger.warning(f"⚠️ No data crawled from {category_type}")
            return {
                "category": category_type,
                "crawled_count": 0,
                "inserted_count": 0,
                "failed_count": 0
            }

        logger.info(f"✅ Crawled {len(crawled_data)} items")

        # Kiểm tra dữ liệu
        with_image = sum(1 for item in crawled_data if item.get("image_url"))
        without_image = len(crawled_data) - with_image
        logger.info(f"📸 Items with image: {with_image}")
        logger.info(f"⚠️ Items without image: {without_image}")

        # Insert vào Milvus theo batch
        logger.info(f"💾 Inserting into Milvus (batch size: {batch_size})...")
        inserted_count = 0
        failed_count = 0

        # Tạo ID duy nhất cho mỗi item
        category_id_offset = self._get_category_id_offset(category_type)

        for i in range(0, len(crawled_data), batch_size):
            batch = crawled_data[i:i + batch_size]

            # Assign unique IDs
            for idx, item in enumerate(batch):
                item["id"] = category_id_offset + i + idx + 1

            try:
                self.dao.insert_data(batch)
                inserted_count += len(batch)
                logger.info(f"  ✅ Inserted batch {i // batch_size + 1}: {len(batch)} items")
            except Exception as e:
                failed_count += len(batch)
                logger.error(f"  ❌ Failed to insert batch {i // batch_size + 1}: {e}")

        stats = {
            "category": category_type,
            "crawled_count": len(crawled_data),
            "inserted_count": inserted_count,
            "failed_count": failed_count,
            "items_with_image": with_image,
            "items_without_image": without_image
        }

        logger.info(f"\n📊 Category '{category_type}' Summary:")
        logger.info(f"  Crawled:  {stats['crawled_count']}")
        logger.info(f"  Inserted: {stats['inserted_count']}")
        logger.info(f"  Failed:   {stats['failed_count']}")
        logger.info(f"  📸 With image: {stats['items_with_image']}")
        logger.info(f"  ⚠️ Without image: {stats['items_without_image']}")

        return stats

    def crawl_all_and_insert(
            self,
            max_items_per_category: int = None,
            max_pages_per_category: int = 10,
            batch_size: int = 10
    ) -> Dict[str, Dict]:
        """
        Crawl tất cả categories và insert vào Milvus

        Args:
            max_items_per_category: Số items tối đa mỗi category
            max_pages_per_category: Số trang tối đa mỗi category
            batch_size: Số items trong mỗi batch insert

        Returns:
            Dict với thống kê cho từng category
        """
        logger.info(f"\n{'=' * 80}")
        logger.info("🌍 CRAWLING AND INSERTING ALL CATEGORIES (OPTIMIZED)")
        logger.info(f"{'=' * 80}")

        all_stats = {}

        for category_type in self.crawler.CATEGORY_URLS.keys():
            try:
                stats = self.crawl_category_and_insert(
                    category_type=category_type,
                    max_items=max_items_per_category,
                    max_pages=max_pages_per_category,
                    batch_size=batch_size
                )
                all_stats[category_type] = stats
            except Exception as e:
                logger.error(f"❌ Error processing {category_type}: {e}")
                all_stats[category_type] = {
                    "category": category_type,
                    "crawled_count": 0,
                    "inserted_count": 0,
                    "failed_count": 0,
                    "error": str(e)
                }

        # Overall summary
        logger.info(f"\n{'=' * 80}")
        logger.info("📊 OVERALL SUMMARY")
        logger.info(f"{'=' * 80}")

        total_crawled = sum(s['crawled_count'] for s in all_stats.values())
        total_inserted = sum(s['inserted_count'] for s in all_stats.values())
        total_failed = sum(s['failed_count'] for s in all_stats.values())
        total_with_image = sum(s.get('items_with_image', 0) for s in all_stats.values())
        total_without_image = sum(s.get('items_without_image', 0) for s in all_stats.values())

        logger.info(f"Total Crawled:  {total_crawled}")
        logger.info(f"Total Inserted: {total_inserted}")
        logger.info(f"Total Failed:   {total_failed}")
        logger.info(f"📸 With image:  {total_with_image} ({total_with_image/total_crawled*100:.1f}%)")
        logger.info(f"⚠️ Without image: {total_without_image} ({total_without_image/total_crawled*100:.1f}%)")

        # Database stats
        db_stats = self.dao.get_statistics()
        logger.info(f"\n💾 Database Statistics:")
        logger.info(f"  Database:   {db_stats['Project']}")
        logger.info(f"  Collection: {db_stats['collection']['name']}")
        logger.info(f"  Total Items: {db_stats['collection']['total_count']}")
        logger.info(f"  Schema: {db_stats['collection']['schema_version']}")

        return all_stats

    def _get_category_id_offset(self, category_type: str) -> int:
        """
        Lấy offset ID cho mỗi category để tránh trùng ID
        Mỗi category có 10000 IDs
        """
        category_offsets = {
            "diem-den": 0,
            "luu-tru": 10000,
            "tour": 20000,
            "nha-hang": 30000,
            "am-thuc": 40000,
            "du-thuyen": 50000
        }
        return category_offsets.get(category_type, 60000)

    def export_stats_to_json(self, stats: Dict, filepath: str = "insert_stats_optimized.json"):
        """Lưu thống kê vào JSON file"""
        logger.info(f"💾 Saving statistics to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Statistics saved!")


def main():
    """Main function"""
    print("=" * 80)
    print("🚀 BÃI CHÁY TOURISM DATA PIPELINE - OPTIMIZED VERSION")
    print("⭐ CHỈ LẤY 1 ẢNH THUMBNAIL VÀ URL BÀI VIẾT")
    print("=" * 80)

    # Cấu hình
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"

    # Tùy chọn crawl
    MAX_ITEMS_PER_CATEGORY = None  # None = crawl tất cả
    MAX_PAGES_PER_CATEGORY = 20  # Số trang tối đa mỗi category
    BATCH_SIZE = 10  # Số items insert mỗi lần

    try:
        # Khởi tạo pipeline
        pipeline = CrawlAndInsertPipeline(
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT
        )

        # Lựa chọn: Crawl một category hay tất cả?
        print("\n📋 Options:")
        print("  1. Crawl and insert ONE category (for testing)")
        print("  2. Crawl and insert ALL categories")

        choice = input("\nYour choice (1 or 2): ").strip()

        if choice == "1":
            # Crawl một category
            print("\n📋 Available categories:")
            categories = list(pipeline.crawler.CATEGORY_URLS.keys())
            for idx, cat in enumerate(categories, 1):
                print(f"  {idx}. {cat}")

            cat_choice = input(f"\nSelect category (1-{len(categories)}): ").strip()
            try:
                cat_idx = int(cat_choice) - 1
                category = categories[cat_idx]

                max_items_input = input(f"\nMax items to crawl (press Enter for all): ").strip()
                max_items = int(max_items_input) if max_items_input else None

                stats = pipeline.crawl_category_and_insert(
                    category_type=category,
                    max_items=max_items,
                    max_pages=MAX_PAGES_PER_CATEGORY,
                    batch_size=BATCH_SIZE
                )

                pipeline.export_stats_to_json({"single_category": stats})

            except (ValueError, IndexError):
                print("❌ Invalid choice!")
                return

        elif choice == "2":
            # Crawl tất cả categories
            all_stats = pipeline.crawl_all_and_insert(
                max_items_per_category=MAX_ITEMS_PER_CATEGORY,
                max_pages_per_category=MAX_PAGES_PER_CATEGORY,
                batch_size=BATCH_SIZE
            )

            pipeline.export_stats_to_json(all_stats)

        else:
            print("❌ Invalid choice!")
            return

        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    main()