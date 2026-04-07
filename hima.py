from fastapi import FastAPI,Form
from fastapi.responses import FileResponse
import pickle

model = pickle.load(open("model.pkl","rb"))
app = FastAPI()

@app.get("/")
def home():
    return FileResponse("prac.html")

@app.post("/predict")
def predict(area: float = Form(...),
            bedrooms: int = Form(...),
            age: int = Form(...)):
    
    price = model.predict([[area, bedrooms, age]])[0]
    return {"prediction": float(price)}