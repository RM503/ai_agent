# Entry-point: to be developed more as features are added
from fastapi import FastAPI

from .common.logging_config import get_logger
from .routers import chat

logger = get_logger(__name__)

app = FastAPI()
app.include_router(chat.chat_router)

