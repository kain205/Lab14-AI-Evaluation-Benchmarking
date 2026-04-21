"""
Tra cứu giá cước Xanh SM từ Dataset/pricedata.json.

Cấu trúc JSON hiện tại:
  priceData.taxi        – Xanh SM Car      {value1, value2}
  priceData.luxury      – Xanh SM Luxury   {value1, value2}  (value2 không có đơn vị)
  priceData.premium     – Xanh SM Premium  {value1, value2}
  priceData.two_ways    – Phụ phí so sánh  {value1, value2, value3, value4}
  (các key mới trong tương lai được tự động nhận diện)
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "pricedata.json")
)

# ---------------------------------------------------------------------------
# Nhãn hiển thị
# ---------------------------------------------------------------------------

SERVICE_LABELS: dict[str, str] = {
    "taxi":      "Xanh SM Car",
    "bike":      "Xanh SM Bike",
    "premium":   "Xanh SM Premium",
    "two_ways":  "Phụ phí",
}

# Alias → key JSON
SERVICE_ALIASES: dict[str, str] = {
    "car":          "taxi",
    "xanh sm car":  "taxi",
    "xanh sm":      "taxi",
    "bike":         "bike",
    "xanh sm bike": "bike",
    "premium":      "premium",
    "xanh sm premium": "premium",
    "phu phi":      "two_ways",
    "phụ phí":      "two_ways",
    "surcharge":    "two_ways",
    "two_ways":     "two_ways",
    "all":          "all",
}

# ---------------------------------------------------------------------------
# Load & index
# ---------------------------------------------------------------------------

_cache: dict | None = None  # {svc_key: {city_lower: {"city": str, "items": list}}}


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    with open(_DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    price_data: dict = raw.get("priceData", {})

    # Build city index per service
    _cache = {}
    for svc_key, svc_data in price_data.items():
        index: dict[str, dict] = {}
        for row in svc_data.get("rows", []):
            city: str = row.get("city", "")
            index[city.lower()] = {
                "city": city,
                "items": row.get("items", []),
                "columns": svc_data.get("columns", ""),
            }
        _cache[svc_key] = index

    logger.info("[FARE] Loaded price data: services=%s", list(_cache.keys()))
    return _cache


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_simple(city: str, label: str, columns: str, items: list) -> str:
    """Định dạng bảng 2 cột (taxi / luxury / premium)."""
    parts = columns.split("|", 1)
    col1 = parts[0].strip() if parts else "Giá dịch vụ"
    col2 = parts[1].strip() if len(parts) > 1 else "Giá niêm yết (VNĐ)"

    lines = [
        f"### {label} – {city}",
        f"| {col1} | {col2} |",
        "|---|---|",
    ]
    for item in items:
        v1 = (item.get("value1") or "").strip()
        v2 = (item.get("value2") or "").strip()
        if v1:
            lines.append(f"| {v1} | {v2} |")

    return "\n".join(lines)


def _fmt_two_ways(city: str, columns: str, items: list) -> str:
    """Định dạng bảng phụ phí đa cột (two_ways)."""
    cols = [c.strip() for c in columns.split("|")]
    n = len(cols)

    header = " | ".join(cols)
    sep    = " | ".join(["---"] * n)
    lines  = [
        f"### Phụ phí – {city}",
        f"| {header} |",
        f"| {sep} |",
    ]

    value_keys = ["value1", "value2", "value3", "value4"][:n]
    for item in items:
        cells = [(item.get(k) or "") for k in value_keys]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_supported_cities() -> list[str]:
    data = _load()
    source = data.get("taxi") or next(iter(data.values()), {})
    return [v["city"] for v in source.values()]


def get_supported_services() -> list[str]:
    return list(_load().keys())


def lookup_fare(city: str, service_type: str = "all") -> str:
    """Tra cứu giá cước theo thành phố và loại dịch vụ."""
    data = _load()

    city_lower = city.strip().lower()
    svc_key = SERVICE_ALIASES.get(service_type.strip().lower(), service_type.strip().lower())

    # Xác định danh sách service cần tra
    if svc_key == "all":
        services = list(data.keys())
    elif svc_key in data:
        services = [svc_key]
    else:
        available = ", ".join(SERVICE_LABELS.get(k, k) for k in data)
        return (
            f"Loại dịch vụ **'{service_type}'** không hợp lệ.\n"
            f"Các loại hiện có: {available}."
        )

    # Thu thập kết quả theo từng service
    sections: list[str] = []
    city_found = False

    for svc in services:
        city_data = data.get(svc, {}).get(city_lower)
        if not city_data:
            continue

        city_found = True
        actual_city = city_data["city"]
        items       = city_data["items"]
        columns     = city_data["columns"]
        label       = SERVICE_LABELS.get(svc, svc.replace("_", " ").title())

        if svc == "two_ways":
            sections.append(_fmt_two_ways(actual_city, columns, items))
        else:
            sections.append(_fmt_simple(actual_city, label, columns, items))

    if not city_found:
        cities = get_supported_cities()
        return (
            f"Không tìm thấy thành phố **'{city}'** trong dữ liệu.\n"
            f"Các thành phố đang hỗ trợ: {', '.join(cities)}."
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# OpenAI tool definition
# ---------------------------------------------------------------------------

FARE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "lookup_fare",
        "description": (
            "Tra cứu bảng giá cước và phụ phí Xanh SM theo thành phố và loại dịch vụ. "
            "Gọi tool này khi người dùng hỏi về: giá cước, chi phí chuyến đi, phụ phí đêm, "
            "giá mở cửa, giá theo km, giờ chờ, thêm điểm đến của Xanh SM tại một thành phố cụ thể."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "Tên thành phố cần tra cứu. "
                        "VD: 'Hà Nội', 'Hà Giang', 'TP. Hồ Chí Minh', 'Đà Nẵng'."
                    ),
                },
                "service_type": {
                    "type": "string",
                    "enum": ["taxi", "bike", "premium", "two_ways", "all"],
                    "description": (
                        "'taxi' – Xanh SM Car (ô tô thường); "
                        "'bike' – Xanh SM Bike; "
                        "'premium' – Xanh SM Premium; "
                        "'two_ways' – bảng phụ phí so sánh Car và Premium; "
                        "'all' – tất cả loại dịch vụ (mặc định)."
                    ),
                },
            },
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> str:
    if name == "lookup_fare":
        return lookup_fare(
            city=args.get("city", ""),
            service_type=args.get("service_type", "all"),
        )
    return f"Tool '{name}' không được hỗ trợ."
