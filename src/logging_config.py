"""
logging_config.py
Centralised Loguru configuration shared by every component (training,
serving, streaming, ingestion, dashboard).

Why this exists: previously each module called `logger.add(...)`
directly with its own ad-hoc rotation settings. That works, but (a) it
duplicates the same six lines everywhere, (b) it makes it easy for one
module's config to drift from another's, and (c) it gives the Streamlit
dashboard no single place to look up "where are this component's logs".

Nothing here changes what gets logged or the scientific pipeline —
it only standardises *how* logs are written and adds a small helper
so the dashboard's "Logs" tab can tail any component's file safely.
"""

from pathlib import Path
from loguru import logger

from src import config

_CONFIGURED: set[str] = set()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Every component that writes logs, used by the dashboard to populate
# a "which log do you want to view" selector.
KNOWN_COMPONENTS = ["api", "consumer", "producer", "train", "feature_engineering", "dashboard"]


def configure_logging(component: str, level: str = None):
    """
    Attaches a rotating file sink (logs/<component>.log) for the given
    component. Idempotent — safe to call more than once per process
    (e.g. on module re-import); the sink is only added the first time.

    Args:
        component: short name used as the log filename, e.g. "api".
        level: minimum log level for this sink. Defaults to
            config.LOG_LEVEL (overridable via the LOG_LEVEL env var).

    Returns:
        The shared loguru `logger` instance, so call sites can write
        `log = configure_logging("api")` and then use `log.info(...)`.
    """
    if component in _CONFIGURED:
        return logger

    level = level or config.LOG_LEVEL
    log_path = config.LOG_DIR / f"{component}.log"

    logger.add(
        log_path,
        rotation="10 MB",
        retention="14 days",
        level=level,
        format=LOG_FORMAT,
        enqueue=True,       # makes writes process/thread-safe
        backtrace=False,    # avoid leaking local variable values in prod logs
        diagnose=False,
    )
    _CONFIGURED.add(component)
    logger.debug(f"Logging configured for component='{component}' -> {log_path}")
    return logger


def tail_log(component: str, n_lines: int = 200) -> str:
    """
    Returns the last `n_lines` of a component's log file as plain text.
    Used by the Streamlit dashboard's "Logs" tab. Never raises — any
    problem reading the file is returned as a human-readable message
    instead, so the dashboard tab never crashes because of a missing
    or locked log file.
    """
    log_path = config.LOG_DIR / f"{component}.log"
    if not log_path.exists():
        return f"(Aucun fichier de log pour « {component} » pour le moment : {log_path})"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:]) if lines else "(fichier de log vide)"
    except Exception as e:
        return f"⚠ Erreur de lecture du log « {component} » : {e}"
