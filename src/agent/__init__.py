from agent.common.logging_config import configure_logging
configure_logging()

from agent.common.logging_config import get_logger
logger = get_logger(__name__)

# Load dotenv once upon package initialization
from dotenv import load_dotenv
load_dotenv()