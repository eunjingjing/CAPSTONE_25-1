from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi import Form 
from ultralytics import YOLO
from recommend import recommend_for_image
import shutil, uuid, os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    handedness: str = Form(...)
):
    image_id = str(uuid.uuid4())
    file_path = f"{image_id}.jpg" 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = recommend_for_image(file_path, handedness=handedness, user_overrides={})
    feedback = result["feedback"]
    score = result["score"]
    os.remove(file_path)

    return JSONResponse(content={
        "feedback": feedback,
        "score": score
    })

