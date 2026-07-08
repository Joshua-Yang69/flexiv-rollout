from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List
import time
from contextlib import contextmanager
import loguru
from loguru import logger
from rich import print


def setup_logger() -> loguru.Logger:
    """
    Setup loguru logger with colored console output and file logging.
    Adapts configuration based on the Operating System.
    """
    # Remove default handler
    logger.remove()
    
    # Detect if current system is Windows
    # sys.platform returns 'win32' on Windows, 'linux' on Linux
    IS_WINDOWS = sys.platform == "win32"
    
    # ------------------------------------------------------------------
    # OS-specific configuration strategy
    # ------------------------------------------------------------------
    if IS_WINDOWS:
        # Windows: Disable rotation and compression to prevent [WinError 32]
        # Although automatic log rotation is disabled, this ensures stability in multi-process scenarios
        rotation_val = None
        retention_val = None
        compression_val = None
    else:
        # Ubuntu/Linux: Enable full features
        # Linux filesystem allows renaming files while open, so rotation is safe
        rotation_val = "1 day"     # Rotate once per day
        retention_val = "7 days"   # Retain for 7 days
        compression_val = "zip"    # Compress old logs

    # ------------------------------------------------------------------
    # 1. Console Handler
    # ------------------------------------------------------------------
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True  # Enable queue on all systems to prevent console output chaos
    )
    
    # Ensure log directory exists
    log_dir = Path(".logs")
    log_dir.mkdir(exist_ok=True)
    
    # ------------------------------------------------------------------
    # 2. File Handler (Debug Log - All logs)
    # ------------------------------------------------------------------
    logger.add(
        log_dir / "debug.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        enqueue=True,         # Must enable: async writing, foundation for multi-process safety
        rotation=rotation_val,       # Dynamically determined based on system
        retention=retention_val,     # Dynamically determined based on system
        compression=compression_val, # Dynamically determined based on system
        backtrace=True,
        diagnose=True
    )
    
    # ------------------------------------------------------------------
    # 3. File Handler (Error Log - Errors only)
    # ------------------------------------------------------------------
    logger.add(
        log_dir / "error.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        enqueue=True,
        rotation=rotation_val,
        retention="30 days" if not IS_WINDOWS else None, # Error logs retained longer on Linux
        compression=compression_val,
        backtrace=True,
        diagnose=True
    )
    
    return logger

setup_logger()


@contextmanager
def timed(section:str, time_dict:Dict[str, List[float]]):
    """Context manager for timing code sections.

    Args:
        section: Name of the section being timed
        time_dict: Dictionary to store timing results
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if section not in time_dict:
        time_dict[section] = [elapsed]
    else:
        time_dict[section].append(elapsed)


__all__ = ["print", "logger", "timed"]


if __name__ == "__main__":
    time_dict = {}
    for _ in range(10):
        logger.debug("before")
        with timed("pre", time_dict):
            time.sleep(0.1) # do something here
        logger.warning("middle")
        with timed("post", time_dict):
            time.sleep(0.1) # do something here
        logger.error(f"after")
    print(time_dict)
