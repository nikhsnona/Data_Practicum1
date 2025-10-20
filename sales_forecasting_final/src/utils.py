from pathlib import Path
import logging

def get_project_root() -> Path:
    # Project root is two levels above this file (sales_forecasting_final/)
    return Path(__file__).resolve().parents[1]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str = "sf_logger") -> logging.Logger:
    root = get_project_root()
    log_dir = root / "logs"
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger
