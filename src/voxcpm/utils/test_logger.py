"""
测试日志工具模块，用于为测试文件提供统一的日志记录功能。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_test_logger(test_file_path: str, log_dir: str = "logs", console_output: bool = True) -> logging.Logger:
    """
    为测试文件设置日志记录器。
    
    Args:
        test_file_path: 测试文件的路径（用于生成日志文件名）
        log_dir: 日志保存目录，默认为 "logs"
    
    Returns:
        配置好的 Logger 实例
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 从测试文件路径提取文件名（不含扩展名）
    test_file_name = Path(test_file_path).stem
    
    # 生成日志文件名：test_streaming_20250101_120000.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{test_file_name}_{timestamp}.log"
    log_filepath = log_path / log_filename
    
    # 创建 logger
    logger = logging.getLogger(test_file_name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 文件 handler：记录所有级别的日志
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 添加文件 handler
    logger.addHandler(file_handler)

    # 添加控制台 handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info(f"日志文件已创建: {log_filepath}")
    
    return logger


def get_test_logger(test_file_path: Optional[str] = None, console_output: bool = False) -> logging.Logger:
    """
    获取测试日志记录器（便捷函数）。
    
    Args:
        test_file_path: 测试文件路径，如果为 None 则从调用栈推断
    
    Returns:
        Logger 实例
    """
    import inspect
    if test_file_path is None:
        # 从调用栈获取测试文件路径
        frame = inspect.currentframe()
        if frame and frame.f_back:
            test_file_path = frame.f_back.f_globals.get('__file__', 'test')
    
    return setup_test_logger(test_file_path or 'test', console_output=console_output)

