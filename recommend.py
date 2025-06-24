import os
import uuid
import cv2
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from ultralytics import YOLO
from typing import List, Tuple, Dict, Set, Optional

# ===================== 설정 및 상수 =====================
CLASS_NAMES = [
    "bag", "books", "bookshelf", "calendar", "correction-tape", "cosmetic",
    "drink", "earphone", "eraser", "food", "gamepad", "glasses", "glue", "goods",
    "headset", "keyboard-pc", "laptop", "mic-pc", "monitor-pc", "mouse-pc", "organizer",
    "paper", "pen", "pen holder", "pencil case", "phone", "photo", "post-it", "ruler",
    "scissors", "snack", "speakers-pc", "stapler", "stopwatch", "tablet-pc", "tape",
    "tissue", "tower-pc", "trash", "watch"
]

GROUPS = {
    "books": ["books", "paper", "post-it"],
    "stationery": ["pen", "pencil case", "scissors", "glue", "tape", "eraser", "stapler", "correction-tape", "pen holder", "ruler"],
    "it": ["laptop", "keyboard-pc", "monitor-pc", "mouse-pc", "tablet-pc", "mic-pc", "headset", "speakers-pc", "tower-pc", "gamepad", "phone", "earphone"],
    "trash": ["food", "drink", "trash", "snack"],
    "personal": ["glasses", "cosmetic", "bag", "watch"],
    "photo": ["photo"],
    "calendar": ["calendar"],
    "goods": ["goods"]
}

STUDY_OBJECTS = {"books", "pen", "ruler", "eraser", "glue", "paper", "post-it", 
                 "tape", "scissors", "stapler", "stopwatch", "tablet-pc", "correction-tape"}
COMPUTER_OBJECTS = {"monitor-pc", "keyboard-pc", "mouse-pc", "laptop", "tablet-pc", 
                    "headset", "mic-pc", "speakers-pc", "tower-pc"}


EXCLUDE_CLASSES_BACKGROUND = {"monitor-pc", "photo", "goods", "post-it"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "classes_weights.csv")
WEIGHTS_DF = pd.read_csv(CSV_PATH)
WEIGHTS_MAP = WEIGHTS_DF.set_index("class").to_dict(orient="index")


# ===================== 유틸 함수 =====================
# === 책상 영역 추출 ===
def get_desk_top_dynamic(
    objs: List[Tuple[int,int,int,int,int]], 
    exclude_classes: Set[str], 
    class_names: List[str], 
    img_h: int,
    consider_ratio: float = 0.3,   # 상위 30% 고려
    margin_ratio: float = 0.05
) -> int:
    cy_list = []
    for (x1, y1, x2, y2, cls_id) in objs:
        label = class_names[cls_id]
        if label in exclude_classes:
            continue
        cy = (y1 + y2) // 2
        cy_list.append(cy)

    if not cy_list:
        raise ValueError("monitor-pc 제외 후 유효 객체 없음")

    # 중심점 y좌표 오름차순 정렬
    sorted_cy = sorted(cy_list)

    # 상위 consider_ratio 비율만 사용
    k = max(1, int(len(sorted_cy) * consider_ratio))
    selected_cy = sorted_cy[:k]

    # 평균으로 top 계산
    avg_cy = int(np.mean(selected_cy))

    margin = int((img_h - avg_cy) * margin_ratio)
    desk_top = max(0, avg_cy - margin)
    return desk_top

# === 책상 그리드(3X4) 정의 및 한글 변환 ===
def create_grid_map(rows:int=3, cols:int=4) -> Tuple[Dict[str, List[Tuple[int,int]]], Dict[str, str]]:
    region_map = {
        "left_top": [(0,0)],
        "top": [(0,1), (0,2)],
        "right_top": [(0,3)],
        "left": [(1,0), (2,0)],
        "right": [(1,3), (2,3)],
        "center": [(1,1), (1,2), (2,1), (2,2)],
    }
    region_kr = {
        "left_top": "좌측 상단",
        "top": "상단",
        "right_top": "우측 상단",
        "left": "왼쪽",
        "right": "오른쪽",
        "center": "중앙"
    }
    return region_map, region_kr

REGION_MAP, REGION_KR = create_grid_map(3, 4)

def get_region_key_from_grid(grid: Tuple[int, int]) -> str:
    for region_key, grids in REGION_MAP.items():
        if grid in grids:
            return region_key
    return "unknown"

def region_to_kr(region:str) -> str:
    return REGION_KR.get(region, region)

def load_and_check_image(image_path:str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    return img

def run_yolo_inference(model, image_path: str, conf_thres: float = 0.45):
    print("🖼️ 이미지 읽는 중...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지 로딩 실패: {image_path}")

    print("⏱️ YOLO 추론 시작 (model(img))")
    t0 = time.time()
    results = model(img, imgsz=320, device='cpu')
    print(f"✅ YOLO 추론 완료 (소요 시간: {time.time() - t0:.2f}초)")

    boxes = results[0].boxes
    objs = [
        (*map(int, box), int(cls_id))
        for box, cls_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf)
        if score >= conf_thres
    ]
    return objs, results

# 객체별 위치를 책상 그리드로 변환
def analyze_objects_by_grid(
    objects:List[Tuple[int,int,int,int,int]], img_h:int, img_w:int, rows:int=3, cols:int=4
) -> Tuple[Dict[Tuple[int,int], List[str]], Dict[str,List[Tuple[int,int]]], List[Tuple[str,Tuple[int,int],Tuple[int,int]]]]:
    desk_top = get_desk_top_dynamic(
        objects, exclude_classes=EXCLUDE_CLASSES_BACKGROUND, class_names=CLASS_NAMES, img_h=img_h
    )
    desk_bottom = img_h
    cell_w, cell_h = img_w / cols, (desk_bottom - desk_top) / rows
    grid_objects, label_grid_map, object_info = defaultdict(list), defaultdict(list), []
    for obj in objects:
        x1, y1, x2, y2, cls_id = obj
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if not (desk_top <= cy <= desk_bottom):
            continue
        grid_r = min(max(int((cy - desk_top) // cell_h), 0), rows-1)
        grid_c = min(max(int(cx // cell_w), 0), cols-1)
        grid, label = (grid_r, grid_c), CLASS_NAMES[cls_id]
        grid_objects[grid].append(label)
        label_grid_map[label].append(grid)
        object_info.append((label, grid, (cx, cy)))
    return grid_objects, label_grid_map, object_info

def compute_recommendations(
    detected_labels: List[str],
    weights_df: pd.DataFrame,
    handedness: str,
    usage: List[str],
    label_grid_map: Dict[str, List[Tuple[int, int]]],
    rows: int = 3,
    cols: int = 4
) -> Dict[str, str]:
    weights_df = weights_df.set_index("class")
    region_objects = {region_key: [] for region_key in REGION_MAP.keys()}

    recommendations = {}

    for label in detected_labels:
        row = weights_df.loc[label]
        if row["base_importance"] == 0:
            recommendations[label] = f"'{label}'은(는) 책상 위에서 치워주세요."
            continue

        # 손잡이 반영: 우측 선호 또는 좌측 선호에 따라 열 가중치 차등
        hand_bias = [0, 0, 0, 0]
        if row["hand_sensitive"]:
            if handedness == "왼손잡이":
                hand_bias = [0, 0.2, 0.5, 1.0]
            elif handedness == "오른손잡이":
                hand_bias = [1.0, 0.5, 0.2, 0]

        # 위치별 점수 계산
        # position_matrix = np.copy(base_position_weight)
        position_matrix = np.zeros((rows, cols))
        for y in range(rows):
                for x in range(cols):
                    # x축
                    x_center_score = 1 - abs(x - 1.5) / 1.5
                    x_side_score = 1 if x in [0, 3] else 0
                    x_weight_norm = (row["x_weight"] - 1) / 2  # 1~3 → 0~1
                    x_score = x_center_score * x_weight_norm + x_side_score * (1 - x_weight_norm)

                    # y축
                    y_bottom_score = y / 2.0
                    y_top_score = 1 if y == 0 else 0
                    y_weight_norm = (row["y_weight"] - 1) / 1  # 1~2 → 0~1
                    y_score = y_bottom_score * y_weight_norm + y_top_score * (1 - y_weight_norm)

                    weight_score = x_score + y_score

                    # 목적에 따른 위치 보너스
                    usage_bonus = 0
                    if "공부 / 취미" in usage and label in STUDY_OBJECTS and y in [1, 2]:
                        usage_bonus += 0.4  # 더 가깝게 배치
                    if "컴퓨터 / 게임" in usage and label in COMPUTER_OBJECTS and y in [1, 2]:
                        usage_bonus += 0.4  # 더 가깝게 배치

                    # 종합 점수 계산
                    score = weight_score    # 일반화 된 가중치
                    score += usage_bonus    # 사용 목적에 따른 가중치
                    score += hand_bias[x]   # 손잡이 가중치

                    base_importance_norm = (row["base_importance"] - 1) / 3  # 1~4 → 0~1
                    # 중심 4셀
                    is_center = (y, x) in [(1,1), (1,2), (2,1), (2,2)]
                    if is_center:
                        score += base_importance_norm
                    else:
                        score += (1 - base_importance_norm)

                    position_matrix[y, x] = score

        best_y, best_x = np.unravel_index(np.argmax(position_matrix), position_matrix.shape)
        best_region_key = get_region_key_from_grid((best_y, best_x))
        best_region_kr = REGION_KR[best_region_key]
        region_objects[best_region_key].append(label)

        # 실제 위치와 비교 후 피드백 제공
        actual_grids = label_grid_map.get(label, [])
        actual_regions = {get_region_key_from_grid(grid) for grid in actual_grids}
        name_kr = row["korean_name"]
        if best_region_key not in actual_regions:
            recommendations[label] = f"'{name_kr}'은(는) 책상 {best_region_kr}에 두는 게 좋아 보여요!"

    return recommendations

# 겹침 정도 계산
def compute_overlap_penalty(boxes: List[List[float]], threshold: float = 0.6) -> int:
    heavy_overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            xi1, yi1 = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
            xi2, yi2 = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            box2_area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > threshold:
                heavy_overlap += 1
    return min(heavy_overlap * 2, 20)

# 점수 산정 함수
def compute_organization_score(
    label_grid_map: Dict[str, List[Tuple[int, int]]],
    boxes: List[List[float]],
    weights_map: Dict[str, dict],
    rows: int = 3,
    cols: int = 4
) -> Tuple[int, Dict[str, int]]:

    score = 100
    breakdown = {}

    # 1. 과다 배치 패널티
    for label, info in weights_map.items():
        max_count = info.get("max_acceptable_count", None)
        over_penalty = info.get("over_count_penalty", 5)
        if max_count is not None:
            count = len(label_grid_map.get(label, []))
            if count > max_count:
                penalty = (count - max_count) * over_penalty
                score -= penalty
                breakdown[f"{label} 과다"] = -penalty

    # 2. 책 분산 감점
    if "books" in label_grid_map and len(set(label_grid_map["books"])) >= 3:
        score -= 20
        breakdown["책 분산"] = -20

    # 5. 그룹 균형 점검 (stationery)
    stationery_labels = [
        "pen", "pencil case", "scissors", "glue", "tape", "eraser", "stapler",
        "correction-tape", "pen holder", "ruler"
    ]
    counter = Counter()
    for label in stationery_labels:
        for grid in label_grid_map.get(label, []):
            counter[grid] += 1
    if len(counter) >= 4:
        score -= 8
        breakdown["문구류 흩어짐"] = -8

    # 6. 겹침 패널티
    overlap_penalty = compute_overlap_penalty(boxes)
    if overlap_penalty > 0:
        score -= overlap_penalty
        breakdown["객체 겹침 감점"] = -overlap_penalty

    return max(score, 0), breakdown

# 책상 그리드 시각화 함수 (동적 책상 영역 할당 기반)
def visualize_desk_grid(
    image_path: str,
    objs: List[Tuple[int, int, int, int, int]],
    rows: int = 3,
    cols: int = 4
) -> str:
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    desk_top = get_desk_top_dynamic(objs, exclude_classes=EXCLUDE_CLASSES_BACKGROUND, class_names=CLASS_NAMES, img_h=h)
    
    # 🔁 FastAPI에서 이미지 저장하는 것과 동일한 방식으로 UUID 이름으로 저장
    image_id = str(uuid.uuid4())
    filename = f"{image_id}.jpg"
    output_path = os.path.join(os.getcwd(), filename)  # 현재 작업 경로 (보통 /workspace)
    
    desk_bottom = h
    cell_w = w // cols
    cell_h = (desk_bottom - desk_top) // rows

    region_cells = {
        "left_top": [(0, 0)],
        "top": [(0, 1), (0, 2)],
        "right_top": [(0, 3)],
        "left": [(1, 0), (2, 0)],
        "right": [(1, 3), (2, 3)],
        "center": [(1, 1), (1, 2), (2, 1), (2, 2)]
    }
    region_colors = {
        "left_top": (255, 204, 255),
        "top": (255, 204, 204),
        "right_top": (255, 255, 204),
        "left": (204, 229, 255),
        "right": (204, 255, 229),
        "center": (255, 255, 255)
    }

    overlay = img.copy()
    alpha = 0.4

    for region, cells in region_cells.items():
        color = region_colors[region]
        for (r, c) in cells:
            pt1 = (c * cell_w, desk_top + r * cell_h)
            pt2 = ((c + 1) * cell_w, desk_top + (r + 1) * cell_h)
            cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for r in range(rows):
        for c in range(cols):
            pt1 = (c * cell_w, desk_top + r * cell_h)
            pt2 = ((c + 1) * cell_w, desk_top + (r + 1) * cell_h)
            cv2.rectangle(img, pt1, pt2, (0, 0, 0), 2)
            region_key = get_region_key_from_grid((c, r))
            label = REGION_KR[region_key]
            cv2.putText(img, label, (pt1[0] + 10, pt1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    cv2.imwrite(output_path, img)
    return output_path

import traceback  # 꼭 필요

def recommend_for_image(image_path: str, handedness: str, user_overrides: dict):
    try:
        MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
        print(f"📦 모델 경로 확인: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("✅ 모델 로딩 완료")

        print("🖼️ 이미지 로딩 시도")
        img = load_and_check_image(image_path)
        print("✅ 이미지 로딩 성공")
        h, w, _ = img.shape
        print(f"📐 이미지 크기: {w} x {h}")

        print("🔎 YOLO 추론 시작")
        objs, results = run_yolo_inference(model, image_path)
        print("✅ YOLO 추론 완료")
        print(f"🔍 탐지된 객체 수: {len(objs)}")

        if not objs:
            print("⚠️ 객체 없음 → 분석 종료")
            return {
                "score": 0,
                "feedback": ["객체가 탐지되지 않았습니다."],
                "image_path": image_path,
                "breakdown": {}
            }

        print("🧭 그리드 분석 중...")
        grid_objects, label_grid_map, object_info = analyze_objects_by_grid(objs, h, w)
        print("✅ 그리드 분석 완료")

        detected_labels = set(label for label, _, _ in object_info)

        # [중요] key 명 확인: app.py에서 "라이프스타일", "사용목적"으로 들어옴
        lifestyle = user_overrides.get("라이프스타일", "")
        usage = user_overrides.get("사용목적", [])

        print(f"👤 사용자 설정 → 라이프스타일: {lifestyle}, 사용목적: {usage}")

        # 추천 위치
        recommendations = compute_recommendations(
            list(detected_labels), WEIGHTS_DF, handedness, usage, label_grid_map
        )
        print(f"🏷️ 탐지된 라벨: {detected_labels}")

        # 정돈 점수 및 감점 breakdown
        score, breakdown = compute_organization_score(label_grid_map, objs, WEIGHTS_MAP, lifestyle)

        # 피드백 구성
        user_feedback = list(recommendations.values())
        custom_feedback = []
        fb_group = []

        # 시각화 이미지 그리드 적용 및 저장
        print("🎨 시각화 이미지 생성 중...")
        grid_img_abspath = visualize_desk_grid(image_path, objs)
        result_img_path = os.path.basename(grid_img_abspath)
        print("✅ 시각화 이미지 저장 완료:", grid_img_abspath)

        # # 시각화(이미지 경로 반환 (EC2가 정적 URL로 렌더링 가능하게))
        # filename = os.path.basename(image_path)
        # result_img_path = f"/static/images/{filename}"

        if not user_feedback:
            user_feedback = ["분석 결과에 따른 피드백이 부족합니다. 입력 설정을 확인해주세요."]
        print("📝 최종 피드백 목록:", user_feedback)
        return {
            "score": score,
            "feedback": list(dict.fromkeys(custom_feedback + user_feedback + fb_group)),    # 현재 user feedback만 사용 중 
            "breakdown": breakdown,
            "image_path": result_img_path
        }

    except Exception as e:
        print("❌ [recommend_for_image] 예외 발생:", str(e))
        traceback.print_exc()
        return {
            "score": 0,
            "feedback": ["분석 중 예외 발생: " + str(e)],
            "breakdown": "error",
            "image_path": ""
        }
