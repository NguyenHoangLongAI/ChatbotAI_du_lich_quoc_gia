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
    """Crawler cho website du lịch Bãi Cháy"""

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
        previous_items_count = 0
        
        # Lấy items từ trang đầu tiên
        first_page_items = self.extract_item_urls_from_list(category_url)
        logger.info(f"    Page 1 has {len(first_page_items)} items")
        
        if not first_page_items:
            logger.warning(f"    Page 1 has no items!")
            return urls
        
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
            
            # Kiểm tra nếu số lượng items giống trang trước (có thể là duplicate page)
            if len(items) == previous_items_count and previous_items_count > 0:
                logger.warning(f"    Page {page_num} has same item count as previous, might be duplicate")
            
            previous_items_count = len(items)
            urls.append(page_url)
            logger.info(f"    Page {page_num} has {len(items)} items")
        
        logger.info(f"  📄 Total pages found: {len(urls)}")
        return urls
        """
        Lấy URLs của tất cả các trang danh sách (pagination)
        Args:
            category_url: URL của category (vd: /diem-du-lich/)
            max_pages: Số trang tối đa để crawl (để tránh vòng lặp vô hạn)
        """
        urls = [category_url]
        
        # Lấy trang đầu tiên để tìm số trang
        soup = self.get_page_soup(category_url)
        if not soup:
            return urls
        
        # Tìm pagination để xác định số trang
        max_page_num = 1
        
        # Tìm tất cả các thẻ <a> có href chứa /page/
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            # Tìm pattern /page/NUMBER/ hoặc /page/NUMBER
            match = re.search(r'/page/(\d+)', href)
            if match:
                page_num = int(match.group(1))
                max_page_num = max(max_page_num, page_num)
                logger.debug(f"    Found page link: {href} -> page {page_num}")
        
        # Nếu không tìm thấy pagination, thử tìm số trong text của các link
        if max_page_num == 1:
            # Tìm các link có text là số
            page_links = soup.find_all('a', href=re.compile(r'/page/'))
            for link in page_links:
                text = link.get_text(strip=True)
                if text.isdigit():
                    page_num = int(text)
                    max_page_num = max(max_page_num, page_num)
        
        # Giới hạn max_pages để tránh crawl quá nhiều
        max_page_num = min(max_page_num, max_pages)
        
        logger.info(f"  📄 Found {max_page_num} pages for this category")
        
        # Tạo URLs cho tất cả các trang
        for page_num in range(2, max_page_num + 1):
            page_url = f"{category_url}page/{page_num}/"
            if page_url not in urls:
                urls.append(page_url)
        
        return urls

    def extract_item_urls_from_list(self, list_url: str) -> List[str]:
        """Trích xuất URLs của các item từ trang danh sách"""
        soup = self.get_page_soup(list_url)
        if not soup:
            return []
        
        item_urls = []
        
        # Tìm khu vực nội dung chính (không lấy sidebar)
        # Thường sidebar có class như 'sidebar', 'widget', hoặc nằm trong aside/sidebar tags
        main_content = soup.find(['main', 'div'], class_=lambda x: x and ('main' in x.lower() or 'content' in x.lower() or 'posts' in x.lower()))
        
        # Nếu không tìm thấy main content, dùng toàn bộ page nhưng loại trừ sidebar
        if not main_content:
            main_content = soup
            # Loại bỏ các phần sidebar/widget
            for sidebar in soup.find_all(['aside', 'div'], class_=lambda x: x and ('sidebar' in x.lower() or 'widget' in x.lower())):
                sidebar.decompose()
        
        # Tìm các link trong các card/item của main content
        items = main_content.find_all(['article', 'div'], class_=lambda x: x and ('post' in x or 'item' in x or 'card' in x))
        
        for item in items:
            link = item.find('a', href=True)
            if link:
                item_url = urljoin(self.BASE_URL, link['href'])
                # Filter: không lấy homepage, login, category pages, sidebar items
                if (item_url not in item_urls and 
                    item_url.startswith(self.BASE_URL) and
                    item_url != self.BASE_URL and
                    item_url != self.BASE_URL + '/' and
                    '/login' not in item_url and
                    '/category/' not in item_url and
                    '/tag/' not in item_url and
                    not item_url.endswith('/diem-du-lich/') and
                    not item_url.endswith('/luu-tru/') and
                    not item_url.endswith('/tour-du-lich/') and
                    not item_url.endswith('/nha-hang/') and
                    not item_url.endswith('/am-thuc/') and
                    not item_url.endswith('/du-thuyen/')):
                    item_urls.append(item_url)
        
        return item_urls

    def extract_detail_info(self, url: str, category_type: str) -> Optional[Dict]:
        """Trích xuất thông tin chi tiết từ trang detail"""
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
            content_divs = soup.find_all(['div', 'section'], class_=lambda x: x and ('content' in x.lower() or 'description' in x.lower()))
            for div in content_divs:
                paragraphs = div.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 20:  # Chỉ lấy đoạn văn có nội dung
                        description_parts.append(text)
            
            description = " ".join(description_parts[:10]) if description_parts else name  # Giới hạn 10 đoạn đầu
            
            # Extract address (FIX: use string= instead of text=)
            address = ""
            address_elem = soup.find(string=re.compile(r'Địa chỉ:|Address:'))
            if address_elem:
                parent = address_elem.find_parent()
                if parent:
                    address = parent.get_text().replace('Địa chỉ:', '').replace('Address:', '').strip()
            
            # Extract price (FIX: use string= instead of text=)
            price_info = {"price_range": "Liên hệ", "price_min": 0.0, "price_max": 0.0}
            price_elem = soup.find(string=re.compile(r'Giá:|Price:|Chi phí:'))
            if price_elem:
                parent = price_elem.find_parent()
                if parent:
                    price_text = parent.get_text()
                    price_info = self.extract_price_info(price_text)
            
            # Extract opening hours (FIX: use string= instead of text=)
            opening_hours = ""
            hours_elem = soup.find(string=re.compile(r'Giờ mở cửa|Opening hours'))
            if hours_elem:
                parent = hours_elem.find_parent()
                if parent:
                    opening_hours = parent.get_text().replace('Giờ mở cửa', '').replace('Opening hours', '').strip()
            
            # Extract images
            image_urls = []
            images = soup.find_all('img', src=True)
            for img in images:
                src = img.get('src', '')
                if src and ('wp-content' in src or 'upload' in src) and not ('logo' in src.lower() or 'icon' in src.lower()):
                    full_url = urljoin(self.BASE_URL, src)
                    if full_url not in image_urls:
                        image_urls.append(full_url)
            
            # Extract sub_type (loại hình)
            sub_type = ""
            type_badge = soup.find(['span', 'a'], class_=lambda x: x and ('category' in x.lower() or 'tag' in x.lower()))
            if type_badge:
                sub_type = type_badge.get_text(strip=True)
            
            # Extract rating (FIX: use string= instead of text=)
            rating = 0.0
            rating_elem = soup.find(string=re.compile(r'☆|★'))
            if rating_elem:
                stars = rating_elem.count('★')
                rating = float(stars) if stars > 0 else 0.0
            
            # Extract view count (FIX: use string= instead of text=)
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
                "image_urls": json.dumps(image_urls[:10], ensure_ascii=False),  # Lưu max 10 ảnh
                "rating": rating,
                "view_count": view_count,
                "url": url
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting detail from {url}: {e}")
            return None

    def crawl_category(self, category_type: str, max_items: int = None, max_pages: int = 10, use_safe_method: bool = True) -> List[Dict]:
        """
        Crawl một category
        Args:
            category_type: diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen
            max_items: Số lượng items tối đa để crawl (None = không giới hạn)
            max_pages: Số trang tối đa để crawl
            use_safe_method: True = crawl từng trang để đảm bảo, False = dùng pagination detection
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Crawling category: {category_type}")
        logger.info(f"{'='*70}")
        
        category_url = urljoin(self.BASE_URL, self.CATEGORY_URLS[category_type])
        
        # Lấy tất cả URLs của các trang danh sách
        if use_safe_method:
            logger.info("  Using safe method: checking each page...")
            list_page_urls = self.get_list_page_urls_safe(category_url, max_pages=max_pages)
        else:
            logger.info("  Using pagination detection method...")
            list_page_urls = self.get_list_page_urls(category_url, max_pages=max_pages)
        
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
        all_item_urls = list(dict.fromkeys(all_item_urls))  # Preserve order while removing duplicates
        
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
                
                item_data["id"] = idx  # Simple ID, có thể dùng hash nếu cần
                results.append(item_data)
                logger.info(f"    ✅ Success")
            else:
                logger.warning(f"    ⚠️ Failed to extract data")
            
            time.sleep(1.5)  # Delay giữa các request
        
        logger.info(f"\n✅ Crawled {len(results)} items from {category_type}")
        return results

    def crawl_all_categories(self, max_items_per_category: int = None, max_pages_per_category: int = 10, use_safe_method: bool = True) -> Dict[str, List[Dict]]:
        """
        Crawl tất cả categories
        Args:
            max_items_per_category: Số items tối đa mỗi category (None = không giới hạn)
            max_pages_per_category: Số trang tối đa mỗi category
            use_safe_method: True = crawl từng trang an toàn, False = dùng pagination detection
        """
        all_data = {}
        
        for category_type in self.CATEGORY_URLS.keys():
            try:
                data = self.crawl_category(
                    category_type, 
                    max_items=max_items_per_category,
                    max_pages=max_pages_per_category,
                    use_safe_method=use_safe_method
                )
                all_data[category_type] = data
            except Exception as e:
                logger.error(f"❌ Error crawling {category_type}: {e}")
                all_data[category_type] = []
        
        return all_data

    def save_to_json(self, data: Dict[str, List[Dict]], filepath: str = "bai_chay_data.json"):
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
    print("Bãi Cháy Tourism Crawler")
    print("=" * 70)
    
    try:
        # Khởi tạo crawler
        crawler = BaiChayCrawler()
        
        # DEBUG: Test extract URLs từ một trang
        print("\n🧪 DEBUG: Testing URL extraction from page 1...")
        test_url = "https://dulichbaichay.vtcnetviet.com/diem-du-lich/"
        urls = crawler.extract_item_urls_from_list(test_url)
        print(f"Found {len(urls)} unique items on page 1:")
        for idx, url in enumerate(urls, 1):
            print(f"  {idx}. {url}")
        
        # Test với page 15 (trong ảnh của bạn)
        print(f"\n🧪 DEBUG: Testing URL extraction from page 15...")
        test_url_15 = "https://dulichbaichay.vtcnetviet.com/diem-du-lich/page/15/"
        urls_15 = crawler.extract_item_urls_from_list(test_url_15)
        print(f"Found {len(urls_15)} items on page 15")
        
        # Test crawl một category với phương pháp an toàn
        print("\n🧪 Testing with 'diem-den' category (safe method, first 10 items)...")
        test_data = crawler.crawl_category(
            "diem-den", 
            max_items=10,  # Giới hạn 10 items để test nhanh
            max_pages=10,
            use_safe_method=True  # Dùng phương pháp an toàn
        )
        
        print(f"\n📊 Crawled {len(test_data)} items")
        if test_data:
            print("\n📝 First 3 items:")
            for idx, item in enumerate(test_data[:3], 1):
                print(f"  {idx}. {item['name']}")
                print(f"     URL: {item['url']}")
        
        print("\n✅ Test completed!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)