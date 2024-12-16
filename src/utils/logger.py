import logging
import sys
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("Rotating Log")
logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout,level=logging.WARN)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
handler = RotatingFileHandler('application.log', maxBytes=20 * 1024 * 1024, backupCount=2)
handler.setFormatter(formatter)
logger.addHandler(handler)