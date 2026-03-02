# Entry-point
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette import status

from routers import uploads
from core.logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.include_router(uploads.router)

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        status_code=status.HTTP_200_OK,
    )

