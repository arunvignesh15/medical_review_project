from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.predict import predict_review

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )
"""

@app.post("/predict") → When the browser sends a POST request to /predict, this function runs.
request: Request → Contains the complete HTTP request (used for rendering the HTML template).
review: str → The text entered by the user.
Form(...) → Tells FastAPI to read the value from the submitted HTML form. The ... means the field is mandatory.

"""

@app.post("/predict")
async def predict(request: Request, review: str = Form(...)):

    prediction, probability = predict_review(review)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "review": review,
            "prediction": prediction,
            "probability": round(probability * 100, 2)
        }
    )