from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pickle


app = FastAPI()

# cors fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods =["*"],
    allow_headers = ["*"]
)

with open("pipeline.pkl",'rb') as f:
    pipeline = pickle.load(f)

class text_data(BaseModel):
    text :str

# home route
@app.get("/")
def home():
    return FileResponse("index.html")

# predction api
@app.post("/prediction")
def predict(data : text_data):
    result = pipeline.predict([data.text])

    return{
        'input':data.text,
        'prediction':result[0]
    }