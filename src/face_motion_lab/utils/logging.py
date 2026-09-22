from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(debug=False):
    folder = Path.home() / ".face-motion-lab"
    folder.mkdir(exist_ok=True)
    handler = RotatingFileHandler(folder / "app.log", maxBytes=2_000_000, backupCount=2)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler, logging.StreamHandler()],
    )
