"""
API 调用跟踪器
提供便捷的装饰器和包装器来自动记录 API 调用
"""

import time
import json
from typing import Dict, Any, Optional, Callable
from functools import wraps
from .logger import APILogger, _default_logger


class APITracker:
    """API 调用跟踪器"""

    def __init__(self, logger: Optional[APILogger] = None):
        self.logger = logger or _default_logger

    def track_request(
        self,
        model: str,
        user: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """装饰器：自动跟踪函数调用并记录日志"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                error = None
                success = True
                response_data = {}

                try:
                    result = func(*args, **kwargs)
                    response_data = result if isinstance(result, dict) else {"result": str(result)}
                    return result
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    request_data = self._extract_request_data(args, kwargs)

                    self.logger.log_call(
                        model=model,
                        request_data=request_data,
                        response_data=response_data,
                        user=user,
                        duration_ms=duration_ms,
                        success=success,
                        error=error,
                        metadata=metadata
                    )

            return wrapper
        return decorator

    def log_completion(
        self,
        model: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        user: str = "default",
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """手动记录一次 API 调用"""
        self.logger.log_call(
            model=model,
            request_data=request_data,
            response_data=response_data,
            user=user,
            duration_ms=duration_ms,
            success=True,
            metadata=metadata
        )

    def wrap_requests_call(
        self,
        model: str,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        user: str = "default",
        verify: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """包装 requests 库的 POST 调用并自动记录"""
        import requests

        start_time = time.time()

        try:
            if payload.get("stream", False):
                if "stream_options" not in payload:
                    payload["stream_options"] = {"include_usage": True}

                response = requests.post(
                    url, headers=headers, json=payload,
                    verify=verify, timeout=timeout, stream=True
                )
                response.raise_for_status()

                return self._handle_stream_response(
                    response, model, payload, user, start_time
                )
            else:
                response = requests.post(
                    url, headers=headers, json=payload,
                    verify=verify, timeout=timeout
                )
                response.raise_for_status()

                duration_ms = (time.time() - start_time) * 1000
                response_data = response.json()

                self.logger.log_call(
                    model=model,
                    request_data=payload,
                    response_data=response_data,
                    user=user,
                    duration_ms=duration_ms,
                    success=True
                )

                return response_data

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.logger.log_call(
                model=model,
                request_data=payload,
                response_data={},
                user=user,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )

            raise

    def _handle_stream_response(
        self,
        response,
        model: str,
        request_data: Dict[str, Any],
        user: str,
        start_time: float
    ):
        """处理流式响应并记录日志"""
        collected_content = []
        collected_usage = None

        def stream_generator():
            nonlocal collected_usage

            try:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        yield line_str + '\n'

                        if line_str.startswith('data: '):
                            data_str = line_str[6:].strip()
                            if data_str != '[DONE]':
                                try:
                                    data = json.loads(data_str)

                                    if "choices" in data:
                                        for choice in data["choices"]:
                                            if "delta" in choice and "content" in choice["delta"]:
                                                content = choice["delta"].get("content")
                                                if content:
                                                    collected_content.append(content)

                                    if "usage" in data and data["usage"] is not None:
                                        collected_usage = data["usage"]

                                except json.JSONDecodeError:
                                    pass
            finally:
                duration_ms = (time.time() - start_time) * 1000
                completion_content = "".join(collected_content)

                response_data = {
                    "content": completion_content,
                    "streaming": True
                }

                if collected_usage:
                    response_data["usage"] = collected_usage

                self.logger.log_call(
                    model=model,
                    request_data=request_data,
                    response_data=response_data,
                    user=user,
                    duration_ms=duration_ms,
                    success=True
                )

        return stream_generator()

    def _extract_request_data(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """从函数参数中提取请求数据"""
        request_data = {}

        for key in ["messages", "prompt", "model", "temperature", "max_tokens"]:
            if key in kwargs:
                request_data[key] = kwargs[key]

        if args and isinstance(args[0], dict):
            request_data.update(args[0])

        return request_data


_default_tracker = APITracker()


def track_request(model: str, user: str = "default", metadata: Optional[Dict[str, Any]] = None):
    """使用默认跟踪器的装饰器"""
    return _default_tracker.track_request(model, user, metadata)


def log_completion(
    model: str,
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    user: str = "default",
    duration_ms: Optional[float] = None
):
    """使用默认跟踪器记录日志"""
    return _default_tracker.log_completion(model, request_data, response_data, user, duration_ms)


def wrap_requests_call(
    model: str,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    user: str = "default",
    **kwargs
):
    """使用默认跟踪器包装 requests 调用"""
    return _default_tracker.wrap_requests_call(model, url, headers, payload, user, **kwargs)
