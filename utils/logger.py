import logging
import os
from datetime import datetime


def setup_logger(
    name="train",
    log_dir="logs",
    level=logging.INFO
):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # ❗ tránh log lặp

    # nếu logger đã có handler → không add nữa
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    print(f"📝 Logging to file: {log_file}")
    return logger
