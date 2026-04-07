from fastapi import FastAPI, Request
from pydantic import BaseModel  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pickle

templates = Jinja2Templates(directory="templates")
app = FastAPI()

# CORS (important if frontend alag run ho raha hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("model.pkl",'rb'))
vectorizer = pickle.load(open("vector.pkl",'rb'))

class input_data(BaseModel):
    text: str


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


@app.post("/prediction")
def predict(data: input_data):
    text = data.text
    
    text_vector = vectorizer.transform([text])
    
    prediction = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    
    return {
        "prediction": int(prediction),
        "probability": float(probs[1])
    }