from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    handedness: str = Form(...),
    lifestyle: str = Form(...),
    purpose: str = Form(...)
):
    # 예시 응답
    return JSONResponse(content={
        "score": 85,
        "feedback": ["좋은 정리 상태입니다!"],
        "image_path": "/static/uploads/sample.jpg"
    })
