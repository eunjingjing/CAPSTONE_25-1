import os
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
        "left_side": [(r,0) for r in range(rows)],
        "right_side": [(r,cols-1) for r in range(rows)],
        "top": [(0,c) for c in range(1,cols-1)],
        "center": [(r,c) for r in range(1,rows) for c in range(1,cols-1)]
    }
    region_kr = {
        "left_side": "왼쪽",
        "right_side": "오른쪽",
        "top": "상단",
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
    lifestyle: str,
    usage: List[str],
    rows: int = 3,
    cols: int = 4
) -> Dict[str, str]:
    weights_df = weights_df.set_index("class")
    region_objects = {region_key: [] for region_key in REGION_MAP.keys()}
    base_position_weight = np.zeros((rows, cols))

    for y in range(rows):
        for x in range(cols):
            x_score = -abs(x - 1.5) + 1.5
            y_score = y
            base_position_weight[y, x] = x_score + y_score

    recommendations = {}

    for label in detected_labels:
        row = weights_df.loc[label]
        if row["base_importance"] == 0:
            recommendations[label] = f"'{label}'은(는) 책상 위에 올려둘 필요가 없어요. 치워주세요!"
            continue

        # 손잡이 반영: 우측 선호 또는 좌측 선호에 따라 열 가중치 차등
        hand_bias = [0, 0, 0, 0]
        if row["hand_sensitive"]:
            if handedness == "왼손잡이":
                hand_bias = [0, 0.2, 0.5, 1.0]
            elif handedness == "오른손잡이":
                hand_bias = [1.0, 0.5, 0.2, 0]

        # 사용 용도 반영
        usage_bonus = 0
        if "공부 / 취미" in usage and row["base_importance"] <= 2:
            usage_bonus = 0.5
        elif "컴퓨터 / 게임" in usage and row["base_importance"] >= 3:
            usage_bonus = 1.0

        # 위치별 점수 계산
        position_matrix = np.copy(base_position_weight)
        for y in range(rows):
            for x in range(cols):
                region_key = get_region_key_from_grid((y, x))
                score = position_matrix[y, x]
                score *= (1 + 0.3 * row["x_weight"] + 0.3 * row["y_weight"])  # 위치 선호 반영
                score += hand_bias[x] + usage_bonus + row["base_importance"]
                clutter_penalty = len(region_objects[region_key]) * (2.0 if lifestyle == "미니멀리스트" else 0.8)
                score -= clutter_penalty
                position_matrix[y, x] = score

        best_y, best_x = np.unravel_index(np.argmax(position_matrix), position_matrix.shape)
        best_region_key = get_region_key_from_grid((best_y, best_x))
        best_region_kr = REGION_KR[best_region_key]
        region_objects[best_region_key].append(label)

        recommendations[label] = f"'{label}'은(는) 책상 {best_region_kr}에 두는 게 좋아 보여요!"

    return recommendations

# 고도화된 정돈 점수 산정 함수: 그룹별 균형, 책 분산, 중요도 우선 배치, 중심 혼잡도, 겹침 등 반영
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

    # 3. 중심 혼잡도 패널티
    center_cells = [(1,1), (1,2), (2,1), (2,2)]
    center_count = sum([
        len(label_grid_map.get(label, []))
        for label in label_grid_map
        for grid in label_grid_map[label]
        if grid in center_cells
    ])
    if center_count >= 6:
        score -= 10
        breakdown["중심 혼잡"] = -10

    # 4. 중요도 높은 객체가 외곽에 있을 경우 감점
    for label, grids in label_grid_map.items():
        if label in weights_map and weights_map[label].get("base_importance", 0) >= 4:
            for grid in grids:
                if grid in [(0,0), (0,3), (2,0), (2,3)]:
                    score -= 5
                    breakdown[f"{label} 위치 부적절"] = -5
                    break

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
    
    # 저장 경로를 자동 생성
    filename = os.path.basename(image_path)
    output_path = os.path.join("/home/ec2-user/my-project/static/images", filename)
    
    desk_bottom = h
    cell_w = w // cols
    cell_h = (desk_bottom - desk_top) // rows

    region_cells = {
        "top": [(0, 1), (0, 2)],
        "left": [(0, 0), (1, 0), (2, 0)],
        "right": [(0, 3), (1, 3), (2, 3)],
        "center": [(1, 1), (1, 2), (2, 1), (2, 2)]
    }
    region_colors = {
        "top": (255, 204, 204),
        "left": (204, 229, 255),
        "right": (204, 255, 229),
        "center": (255, 255, 204)
    }

    # 각 셀 채우기
    for region, cells in region_cells.items():
        for (r, c) in cells:
            pt1 = (c * cell_w, desk_top + r * cell_h)
            pt2 = ((c + 1) * cell_w, desk_top + (r + 1) * cell_h)
            cv2.rectangle(img, pt1, pt2, region_colors[region], thickness=-1)

    # 셀 테두리 + 라벨
    for r in range(rows):
        for c in range(cols):
            pt1 = (c * cell_w, desk_top + r * cell_h)
            pt2 = ((c + 1) * cell_w, desk_top + (r + 1) * cell_h)
            cv2.rectangle(img, pt1, pt2, (0, 0, 0), 2)
            label = f"[{r},{c}]"
            cv2.putText(img, label, (pt1[0] + 10, pt1[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

    cv2.imwrite(output_path, img)
    return output_path

def recommend_for_image(image_path: str, handedness: str, user_overrides: dict):
    try:
        MODEL_PATH = os.path.join(BASE_DIR, "models/weights/best.pt")
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
            return {"score": 0, "feedback": ["객체가 탐지되지 않았습니다."], "image_path": image_path}

        print("🧭 그리드 분석 중...")
        grid_objects, label_grid_map, object_info = analyze_objects_by_grid(objs, h, w)
        print("✅ 그리드 분석 완료")

        detected_labels = set(label for label, _, _ in object_info)
        # 추천 배치 위치
        # 사용자 설정값 적용
        lifestyle = user_overrides.get("lifestyle", "")  # ex: "미니멀리스트"
        usage = user_overrides.get("usage", "")  # "공부 / 취미,컴퓨터 / 게임" 같은 입력

        recommendations = compute_recommendations(
            list(detected_labels), WEIGHTS_DF, handedness, lifestyle, usage
        )
        print(f"🏷️ 탐지된 라벨: {detected_labels}")
        
        # 정돈 점수 및 감점 breakdown
        score, breakdown = compute_organization_score(label_grid_map, objs, WEIGHTS_MAP)

        # 피드백 (기본은 추천 메시지로 대체)
        user_feedback = list(recommendations.values())
        custom_feedback = []  # ← 이후 커스텀 룰 기반 피드백 함수 연결 예정이라면 여기에
        fb_group = []         # ← 그룹 피드백 추후 확장

        # 시각화
        result_img_path = visualize_desk_grid(image_path=image_path, objs=objs)
        return {
            "score": score,
            "feedback": list(dict.fromkeys(custom_feedback + user_feedback + fb_group)),
            "breakdown": breakdown,
            "image_path": result_img_path
        }

    except Exception as e:
        print("❌ [recommend_for_image] 오류 발생:", str(e))
        return {
            "score": 0,
            "feedback": [],
            "breakdown": "error",
            "image_path": ""
        }
