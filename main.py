from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from recommend import recommend_for_image
import shutil, uuid, os, base64

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    handedness: str = Form(...),
    lifestyle: str = Form(...),
    purpose: str = Form(...)
):
    # 1. 이미지 저장
    image_id = str(uuid.uuid4())
    file_path = f"{image_id}.jpg"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 분석 수행
    result = recommend_for_image(
        file_path,
        handedness=handedness,
        user_overrides={
            "라이프스타일": lifestyle,
            "사용목적": purpose.split(',') if purpose else []
        }
    )
    feedback = result["feedback"]
    score = result["score"]
    breakdown = result.get("breakdown", {})
    result_img_path = result.get("image_path", "")

    # 3. 시각화 이미지 → base64 인코딩
    encoded_img = ""
    image_filename = os.path.basename(result_img_path)
    if result_img_path and os.path.exists(result_img_path):
        with open(result_img_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode("utf-8")

    # 4. 임시 업로드 이미지 삭제
    os.remove(file_path)

    # 5. 응답 반환
    return JSONResponse(content={
        "score": score,
        "feedback": feedback,
        "breakdown": breakdown,
        "image_base64": encoded_img,
        "image_filename": image_filename
    })
