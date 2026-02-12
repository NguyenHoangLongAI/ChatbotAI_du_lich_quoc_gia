import requests
from bs4 import BeautifulSoup
import json
import re
import time
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaiChayCrawler:
    """Crawler tối ưu cho website du lịch Bãi Cháy - chỉ lấy 1 ảnh thumbnail"""

    BASE_URL = "https://dulichbaichay.vtcnetviet.com"

    CATEGORY_URLS = {
        "diem-den": "/diem-du-lich/",
        "luu-tru": "/luu-tru/",
        "tour": "/tour-du-lich/",
        "nha-hang": "/nha-hang/",
        "am-thuc": "/am-thuc/",
        "du-thuyen": "/du-thuyen/"
    }

    def __init__(self, embedding_model: str = "keepitreal/vietnamese-sbert"):
        """
        Args:
            embedding_model: Model để tạo embeddings (768 dim)
        """
        logger.info(f"🔄 Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_page_soup(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Lấy BeautifulSoup object từ URL"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"❌ Failed to fetch {url}")
                    return None

    def extract_price_info(self, text: str) -> Dict:
        """
        Trích xuất thông tin giá từ text
        Returns: {"price_range": str, "price_min": float, "price_max": float}
        """
        if not text or "miễn phí" in text.lower():
            return {"price_range": "Miễn phí", "price_min": 0.0, "price_max": 0.0}

        # Tìm các số trong text
        numbers = re.findall(r'[\d,.]+', text.replace('.', '').replace(',', ''))
        prices = [float(n) for n in numbers if n]

        if not prices:
            return {"price_range": "Liên hệ", "price_min": 0.0, "price_max": 0.0}

        price_min = min(prices)
        price_max = max(prices)

        return {
            "price_range": text.strip(),
            "price_min": price_min,
            "price_max": price_max if price_max > price_min else price_min
        }

    def get_list_page_urls_safe(self, category_url: str, max_pages: int = 10) -> List[str]:
        """
        Phương pháp an toàn: Crawl từng trang cho đến khi không còn items
        Args:
            category_url: URL của category
            max_pages: Số trang tối đa để thử
        """
        urls = [category_url]

        # Thử crawl từng trang
        for page_num in range(2, max_pages + 1):
            page_url = f"{category_url}page/{page_num}/"

            # Kiểm tra xem trang có tồn tại không
            soup = self.get_page_soup(page_url)
            if not soup:
                logger.info(f"    Page {page_num} not found, stopping at page {page_num - 1}")
                break

            # Kiểm tra xem có items không
            items = self.extract_item_urls_from_list(page_url)
            if not items:
                logger.info(f"    Page {page_num} has no items, stopping at page {page_num - 1}")
                break

            urls.append(page_url)
            logger.info(f"    Found page {page_num} with {len(items)} items")

        logger.info(f"  📄 Total pages found: {len(urls)}")
        return urls

    def extract_item_urls_from_list(self, list_url: str) -> List[str]:
        """Trích xuất URLs của các item từ trang danh sách"""
        soup = self.get_page_soup(list_url)
        if not soup:
            return []

        item_urls = []

        # Tìm các link trong các card/item
        items = soup.find_all(['article', 'div'], class_=lambda x: x and ('post' in x or 'item' in x or 'card' in x))

        for item in items:
            link = item.find('a', href=True)
            if link:
                item_url = urljoin(self.BASE_URL, link['href'])
                # Filter: không lấy homepage, login, category pages
                if (item_url not in item_urls and
                        item_url.startswith(self.BASE_URL) and
                        item_url != self.BASE_URL and
                        item_url != self.BASE_URL + '/' and
                        '/login' not in item_url and
                        '/category/' not in item_url and
                        not item_url.endswith('/diem-du-lich/') and
                        not item_url.endswith('/luu-tru/') and
                        not item_url.endswith('/tour-du-lich/') and
                        not item_url.endswith('/nha-hang/') and
                        not item_url.endswith('/am-thuc/') and
                        not item_url.endswith('/du-thuyen/')):
                    item_urls.append(item_url)

        return item_urls

    def extract_thumbnail_image(self, soup: BeautifulSoup, url: str) -> str:
        """
        Trích xuất ĐÚNG 1 ảnh thumbnail/featured image từ bài viết
        Ưu tiên: featured image > first content image > og:image
        """
        # 1. Thử tìm featured image (ảnh nổi bật của bài viết)
        featured_img = soup.find('img', class_=re.compile(r'featured|thumbnail|wp-post-image', re.I))
        if featured_img and featured_img.get('src'):
            img_url = featured_img.get('src')
            if self._is_valid_image(img_url):
                return urljoin(self.BASE_URL, img_url)

        # 2. Thử tìm ảnh đầu tiên trong nội dung bài viết
        content_divs = soup.find_all(['div', 'article'],
                                     class_=lambda x: x and ('content' in x.lower() or 'entry' in x.lower()))
        for div in content_divs:
            img = div.find('img', src=True)
            if img:
                img_url = img.get('src')
                if self._is_valid_image(img_url):
                    return urljoin(self.BASE_URL, img_url)

        # 3. Fallback: og:image meta tag
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            img_url = og_image.get('content')
            if self._is_valid_image(img_url):
                return urljoin(self.BASE_URL, img_url)

        # 4. Fallback cuối: ảnh đầu tiên trong toàn bộ trang (ngoại trừ logo/icon)
        all_images = soup.find_all('img', src=True)
        for img in all_images:
            img_url = img.get('src', '')
            if self._is_valid_image(img_url):
                return urljoin(self.BASE_URL, img_url)

        # Không tìm thấy ảnh hợp lệ
        return ""

    def _is_valid_image(self, img_url: str) -> bool:
        """Kiểm tra ảnh có hợp lệ không"""
        if not img_url:
            return False

        # Loại bỏ ảnh placeholder/lazy load
        if 'data:image' in img_url or 'placeholder' in img_url.lower():
            return False

        # Loại bỏ logo, icon, avatar
        exclude_keywords = ['logo', 'icon', 'avatar', 'user-', 'flag', 'banner-footer']
        if any(keyword in img_url.lower() for keyword in exclude_keywords):
            return False

        # Chỉ lấy ảnh từ wp-content (ảnh thật của bài viết)
        if 'wp-content' in img_url or 'upload' in img_url:
            return True

        return False

    def extract_detail_info(self, url: str, category_type: str) -> Optional[Dict]:
        """Trích xuất thông tin chi tiết từ trang detail - CHỈ 1 ẢNH"""
        soup = self.get_page_soup(url)
        if not soup:
            return None

        try:
            # Extract title/name
            title = soup.find('h1')
            name = title.get_text(strip=True) if title else "Unknown"

            # Extract description
            description_parts = []

            # Tìm phần thông tin chung
            content_divs = soup.find_all(['div', 'section'],
                                         class_=lambda x: x and ('content' in x.lower() or 'description' in x.lower()))
            for div in content_divs:
                paragraphs = div.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 20:  # Chỉ lấy đoạn văn có nội dung
                        description_parts.append(text)

            description = " ".join(description_parts[:10]) if description_parts else name  # Giới hạn 10 đoạn đầu

            # Extract address
            address = ""
            address_elem = soup.find(string=re.compile(r'Địa chỉ:|Address:'))
            if address_elem:
                parent = address_elem.find_parent()
                if parent:
                    address = parent.get_text().replace('Địa chỉ:', '').replace('Address:', '').strip()

            # Extract price
            price_info = {"price_range": "Liên hệ", "price_min": 0.0, "price_max": 0.0}
            price_elem = soup.find(string=re.compile(r'Giá:|Price:|Chi phí:'))
            if price_elem:
                parent = price_elem.find_parent()
                if parent:
                    price_text = parent.get_text()
                    price_info = self.extract_price_info(price_text)

            # Extract opening hours
            opening_hours = ""
            hours_elem = soup.find(string=re.compile(r'Giờ mở cửa|Opening hours'))
            if hours_elem:
                parent = hours_elem.find_parent()
                if parent:
                    opening_hours = parent.get_text().replace('Giờ mở cửa', '').replace('Opening hours', '').strip()

            # ⭐ CHỈ LẤY 1 ẢNH THUMBNAIL
            thumbnail_url = self.extract_thumbnail_image(soup, url)

            # Extract sub_type (loại hình)
            sub_type = ""
            type_badge = soup.find(['span', 'a'],
                                   class_=lambda x: x and ('category' in x.lower() or 'tag' in x.lower()))
            if type_badge:
                sub_type = type_badge.get_text(strip=True)

            # Extract rating
            rating = 0.0
            rating_elem = soup.find(string=re.compile(r'☆|★'))
            if rating_elem:
                stars = rating_elem.count('★')
                rating = float(stars) if stars > 0 else 0.0

            # Extract view count
            view_count = 0
            view_elem = soup.find(string=re.compile(r'Lượt xem|Views'))
            if view_elem:
                view_match = re.search(r'(\d+)', view_elem)
                if view_match:
                    view_count = int(view_match.group(1))

            return {
                "name": name,
                "type": category_type,
                "sub_type": sub_type,
                "location": "Bãi Cháy, Quảng Ninh",
                "address": address,
                "description": description[:4900],  # Giới hạn để fit vào VARCHAR(5000)
                "price_range": price_info["price_range"],
                "price_min": price_info["price_min"],
                "price_max": price_info["price_max"],
                "opening_hours": opening_hours,
                "image_url": thumbnail_url,  # ⭐ CHỈ 1 URL DUY NHẤT
                "rating": rating,
                "view_count": view_count,
                "url": url  # ⭐ URL BÀI VIẾT
            }

        except Exception as e:
            logger.error(f"❌ Error extracting detail from {url}: {e}")
            return None

    def crawl_category(self, category_type: str, max_items: int = None, max_pages: int = 10) -> List[Dict]:
        """
        Crawl một category - CHỈ LẤY 1 ẢNH VÀ URL BÀI VIẾT
        Args:
            category_type: diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen
            max_items: Số lượng items tối đa để crawl (None = không giới hạn)
            max_pages: Số trang tối đa để crawl
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"🚀 Crawling category: {category_type}")
        logger.info(f"{'=' * 70}")

        category_url = urljoin(self.BASE_URL, self.CATEGORY_URLS[category_type])

        # Lấy tất cả URLs của các trang danh sách
        logger.info("  Using safe method: checking each page...")
        list_page_urls = self.get_list_page_urls_safe(category_url, max_pages=max_pages)

        logger.info(f"📄 Will crawl {len(list_page_urls)} pages")

        # Lấy URLs của tất cả items
        all_item_urls = []
        for page_idx, list_url in enumerate(list_page_urls, 1):
            logger.info(f"  📋 [{page_idx}/{len(list_page_urls)}] Fetching items from: {list_url}")
            item_urls = self.extract_item_urls_from_list(list_url)
            logger.info(f"     Found {len(item_urls)} items on this page")
            all_item_urls.extend(item_urls)

            # Break nếu đã đủ items
            if max_items and len(all_item_urls) >= max_items:
                logger.info(f"     ✅ Reached max_items limit ({max_items})")
                break

            time.sleep(1)  # Delay giữa các request

        # Loại bỏ duplicate URLs
        all_item_urls = list(dict.fromkeys(all_item_urls))

        # Cắt theo max_items nếu cần
        if max_items:
            all_item_urls = all_item_urls[:max_items]

        logger.info(f"✅ Total unique items to crawl: {len(all_item_urls)}")

        # Crawl chi tiết từng item
        results = []
        for idx, item_url in enumerate(all_item_urls, 1):
            logger.info(f"  [{idx}/{len(all_item_urls)}] Crawling: {item_url}")

            item_data = self.extract_detail_info(item_url, category_type)
            if item_data:
                # Tạo embedding cho description
                logger.info(f"    🔄 Generating embedding...")
                item_data["description_vector"] = self.model.encode(
                    item_data["description"]
                ).tolist()

                item_data["id"] = idx  # Simple ID
                results.append(item_data)

                # Log thông tin ảnh
                if item_data.get("image_url"):
                    logger.info(f"    📸 Thumbnail: {item_data['image_url'][:80]}...")
                else:
                    logger.info(f"    ⚠️ No thumbnail found")

                logger.info(f"    ✅ Success")
            else:
                logger.warning(f"    ⚠️ Failed to extract data")

            time.sleep(1.5)  # Delay giữa các request

        logger.info(f"\n✅ Crawled {len(results)} items from {category_type}")
        return results

    def crawl_all_categories(self, max_items_per_category: int = None, max_pages_per_category: int = 10) -> Dict[
        str, List[Dict]]:
        """
        Crawl tất cả categories
        Args:
            max_items_per_category: Số items tối đa mỗi category (None = không giới hạn)
            max_pages_per_category: Số trang tối đa mỗi category
        """
        all_data = {}

        for category_type in self.CATEGORY_URLS.keys():
            try:
                data = self.crawl_category(
                    category_type,
                    max_items=max_items_per_category,
                    max_pages=max_pages_per_category
                )
                all_data[category_type] = data
            except Exception as e:
                logger.error(f"❌ Error crawling {category_type}: {e}")
                all_data[category_type] = []

        return all_data

    def save_to_json(self, data: Dict[str, List[Dict]], filepath: str = "bai_chay_data_optimized.json"):
        """Lưu dữ liệu crawl vào JSON file"""
        logger.info(f"💾 Saving data to {filepath}...")

        # Remove vectors for JSON export (too large)
        data_without_vectors = {}
        for category, items in data.items():
            data_without_vectors[category] = [
                {k: v for k, v in item.items() if k != "description_vector"}
                for item in items
            ]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_without_vectors, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Data saved to {filepath}")


if __name__ == "__main__":
    print("=" * 70)
    print("Bãi Cháy Tourism Crawler - OPTIMIZED (1 Image Only)")
    print("=" * 70)

    try:
        # Khởi tạo crawler
        crawler = BaiChayCrawler()

        # Test với 1 category
        print("\n🧪 Testing with 'diem-den' category (first 5 items)...")
        test_data = crawler.crawl_category(
            "diem-den",
            max_items=5,
            max_pages=2
        )

        print(f"\n📊 Crawled {len(test_data)} items")
        if test_data:
            print("\n📝 Sample item:")
            sample = test_data[0]
            print(f"  Name: {sample['name']}")
            print(f"  URL: {sample['url']}")
            print(f"  Image: {sample['image_url']}")

        print("\n✅ Test completed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)