"""
UI Route Controller.
Renders the Jinja2 template and serves static UI assets.
"""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

ui_router = APIRouter(include_in_schema=False)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@ui_router.get("/", response_class=HTMLResponse)
@ui_router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    """Render the interactive RAG playground and evaluation dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})
