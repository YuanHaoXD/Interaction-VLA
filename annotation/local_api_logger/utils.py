"""
工具函数模块
提供 token 估算、数据处理等辅助功能
"""


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量"""
    if not text:
        return 0

    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars

    estimated_tokens = (chinese_chars / 2) + (other_chars / 4)

    return int(estimated_tokens)


def format_token_count(count: int) -> str:
    """格式化 token 数量显示"""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> float:
    """计算调用成本"""
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def extract_model_name(model: str) -> str:
    """从完整模型名称中提取简短名称"""
    parts = model.split('-')
    if len(parts) > 2 and parts[-1].isdigit():
        return '-'.join(parts[:-3]) if len(parts) > 3 else model
    return model


def safe_get(d: dict, *keys, default=None):
    """安全地从嵌套字典中获取值"""
    result = d
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
