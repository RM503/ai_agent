# Entry-point: to be developed more as features are added
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette import status

from .routers import transcription
from .core.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="agent/static"), name="static")
templates = Jinja2Templates(directory="agent/templates")

# Imported routes from agent/routers
app.include_router(transcription.page_router)
app.include_router(transcription.api_router)

# Index page
@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        status_code=status.HTTP_200_OK,
    )
