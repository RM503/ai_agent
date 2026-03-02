# Entry-point: to be developed more as features are added
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette import status

from .routers import transcription
from .core.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="agent/templates")

# Imported routes from agent/routers
app.include_router(transcription.router)

# Index page
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        status_code=status.HTTP_200_OK,
    )

