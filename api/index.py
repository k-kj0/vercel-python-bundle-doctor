from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from core.analyzer import analyze_requirements

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})

@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, requirements_text: str = Form(...)):
    result = await analyze_requirements(requirements_text)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"result": result, "input_text": requirements_text},
    )
