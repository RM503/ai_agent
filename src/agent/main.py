# Entry-point: to be developed more as features are added
from fastapi import FastAPI

from .common.logging_config import get_logger
from .routers import auth, chat, upload

logger = get_logger(__name__)

app = FastAPI()
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(upload.router)