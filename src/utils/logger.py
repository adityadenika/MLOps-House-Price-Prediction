import json, logging, sys
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage()
        }
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log.update(record.extra)  # structured fields
        return json.dumps(log)

def get_logger(name: str = "app"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    logger.propagate = False
    return logger
